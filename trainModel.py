import numpy as np
from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset

from modelResults import Results
from lstmEvaluation import evaluate_lstm_model, sample_lstm_hyperparameters

NUM_SAMPLES = 100


def run(train, test, validation, random_state, model_type):
    results = Results('{}_results.txt'.format(model_type), best_metric='test_NDCG10')
    best_result = results.best()
    if model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    for hyperparameters in sample_fnc(random_state, NUM_SAMPLES):
        if hyperparameters in results:
            continue
        print('Evaluating {}'.format(hyperparameters))
        test_metrics, val_metrics = eval_fnc(hyperparameters,
                                             train,
                                             test,
                                             validation,
                                             random_state)
        print('Test metrics:')
        print(test_metrics)
        print('Validation metrics:')
        print(val_metrics)
        results.save(hyperparameters, test_metrics, val_metrics)
    return results


if __name__ == '__main__':
    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('1M')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length,
                                        step_size=step_size)

    mode = 'lstm'
    run(train, test, validation, random_state, mode)
