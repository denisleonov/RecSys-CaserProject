from tqdm.auto import tqdm
import numpy as np
import scipy.stats as st
from sklearn.metrics import ndcg_score


FLOAT_MIN = np.finfo(np.float32).min
FLOAT_MAX = np.finfo(np.float32).max


def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """
    
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()

    test_pbar = tqdm(enumerate(test), total=test.shape[0], desc='Eval', leave=True)
    
    for user_id, row in test_pbar:

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, mean_aps


def sequence_hr_score(model, test, k=(1, 5, 10), exclude_preceding=True):
    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    hrs = [[] for _ in k]
    for seq, target in zip(sequences, targets):
        predictions = -model.predict(seq)

        if exclude_preceding:
            predictions[seq] = FLOAT_MAX

        target_ranks = st.rankdata(predictions, method='min')[target]
        for i in range(len(k)):
            hrs[i].append((target_ranks <= k[i]).mean())
    return np.array(hrs)


def sequence_ndcg_score(model, test, k=(1, 5, 10), exclude_preceding=True):
    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    ndcgs = [[] for _ in k]
    for seq, target in zip(sequences, targets):
        predictions = model.predict(seq)
        ground_truth = np.zeros_like(predictions)
        ground_truth[target] = 1
        if exclude_preceding:
            predictions[seq] = FLOAT_MIN
        for i in range(len(k)):
            ndcgs[i].append(ndcg_score([ground_truth], [predictions], k=k[i]))
    return np.array(ndcgs)
