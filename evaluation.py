import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score


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
    ndcgs = [list() for _ in range(len(ks))]
    apks = list()

    test_pbar = tqdm(enumerate(test), total=test.shape[0], desc='Eval')
    
    for user_id, row in test_pbar:

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]
        predictions = np.array(predictions)
        predictions_sorted_list = list(predictions.argsort())

        targets_list = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets_list, predictions_sorted_list, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)
            
            targets = np.zeros_like(predictions)
            targets[targets_list] = 1.
            
            prd = predictions[np.newaxis, :]
            tgt = targets[np.newaxis, :]
            
            ndcg = ndcg_score(tgt, prd, k=_k)
            ndcgs[i].append(ndcg)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]
    ndcgs = [np.array(i) for i in ndcgs]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]
        ndcgs = ndcgs[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, ndcgs, mean_aps
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
    ndcgs = [list() for _ in range(len(ks))]
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
            # ndcg = ndcg_score(targets, predictions, k=_k)
            ndcg = 0.
            ndcgs[i].append(ndcg)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]
    ndcgs = [np.array(i) for i in ndcgs]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]
        ndcgs = ndcgs[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, ndcgs, mean_aps

