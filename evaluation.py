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

    test_pbar = tqdm(enumerate(test), total=test.shape[0], desc='Eval / Precision and Recall', leave=True)

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


def compute_metrics(model, test, train=None, k=10):
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    ndcgs = [list() for _ in range(len(ks))]
    hrs = [list() for _ in range(len(ks))]
    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = []
    mrrs = []

    test_pbar = tqdm(enumerate(test), total=test.shape[0], desc='Eval', leave=True)

    for user_id, row in test_pbar:

        if not len(row.indices):
            continue

        predictions = model.predict(user_id)
        targets = row.indices

        predictions = np.array(predictions)

        if train is not None:
            # P and R:
            predictions_pr = np.copy(predictions)
            predictions_pr = (-predictions_pr).argsort()
            rated = set(train[user_id].indices)
            predictions_pr = [p for p in predictions_pr if p not in rated]
            # NDCG
            predictions_ndcg = np.copy(predictions)
            predictions_ndcg[train[user_id].indices] = FLOAT_MIN
            # HR
            predictions_hr = np.copy(predictions)
            predictions_hr = - predictions_hr
            predictions_hr[train[user_id].indices] = FLOAT_MAX
            # MRR
            predictions_mrr = np.copy(predictions)
            predictions_mrr = - predictions_mrr
            predictions_mrr[train[user_id].indices] = FLOAT_MAX

        # NDCG
        ground_truth = np.zeros_like(predictions)
        ground_truth[targets] = 1
        # HR
        target_ranks = st.rankdata(predictions_hr, method='min')[targets]  # noqa

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions_pr, _k) # noqa
            precisions[i].append(precision)
            recalls[i].append(recall)
            ndcgs[i].append(ndcg_score([ground_truth], [predictions_ndcg], k=_k)) # noqa
            hrs[i].append((target_ranks <= _k).mean())

        mrrs.append((1.0 / st.rankdata(predictions_mrr)[targets]).mean()) # noqa
        apks.append(_compute_apk(targets, predictions_pr, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]
    ndcgs = [np.array(i) for i in ndcgs]
    hrs = [np.array(i) for i in hrs]

    if not isinstance(k, list):
        ndcgs = ndcgs[0]
        hrs = hrs[0]
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)
    mrr = np.mean(mrrs)

    return precisions, recalls, mean_aps, ndcgs, hrs, mrr
