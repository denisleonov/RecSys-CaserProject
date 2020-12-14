import numpy as np
import scipy.stats as st
from sklearn.metrics import ndcg_score


def sequence_mrr_score(predictions, targets):
    mrrs = []
    for prediction, target in zip(predictions, targets):
        mrr = (1.0 / st.rankdata(-prediction)[target]).mean()
        mrrs.append(mrr)
    return np.array(mrrs)


def sequence_hr_score(predictions, targets, k=[10]):
    hrs = [[] for _ in k]
    for prediction, target in zip(predictions, targets):
        target_ranks = st.rankdata(-prediction, method='min')[target]
        for i in range(len(k)):
            hrs[i].append((target_ranks <= k[i]).mean())
    return np.array(hrs)


def sequence_ndcg_score(predictions, targets, k=[10]):
    ndcgs = [[] for _ in k]
    for prediction, target in zip(predictions, targets):
        ground_truth = np.zeros_like(prediction)
        ground_truth[target] = 1
        for i in range(len(k)):
            ndcgs[i].append(ndcg_score([ground_truth], [prediction], k=k[i]))
    return np.array(ndcgs)
