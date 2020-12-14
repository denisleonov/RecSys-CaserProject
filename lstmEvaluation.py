import os
import shutil

from collections import namedtuple

from sklearn.model_selection import ParameterSampler
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel

from metricsEvaluator import sequence_ndcg_score, sequence_hr_score

CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [256]
EMBEDDING_DIM = [64, 128, 256]
N_ITER = [20]
L2 = [1e-5, 1e-3, 1e-2, 0.0, 0.1]

# LEARNING_RATES = [1e-3]
# LOSSES = ['pointwise']
# BATCH_SIZE = [256]
# EMBEDDING_DIM = [128]
# N_ITER = [20]
# L2 = [1e-3]

K = [5, 10, 25]
Metrics = namedtuple('Metrics', ('MRR', *[f'NDCG{k}' for k in K],
                                 *[f'HR{k}' for k in K]))


def sample_lstm_hyperparameters(random_state, num):
    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        yield params


def evaluate_lstm_model(hyperparameters, train, test, validation, random_state):
    h = hyperparameters
    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)
    model.fit(train, verbose=True)

    test_metrics = evaluate_metrics(model, test)
    val_metrics = evaluate_metrics(model, validation)

    return test_metrics, val_metrics


def evaluate_metrics(model, test):
    metrics = [sequence_mrr_score(model, test).mean()]
    metrics.extend(sequence_ndcg_score(model, test, K).mean(axis=1))
    metrics.extend(sequence_hr_score(model, test, K).mean(axis=1))
    return Metrics(*metrics)
