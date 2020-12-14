import numpy as np
import scipy.stats as st
from sklearn.metrics import ndcg_score

FLOAT_MIN = np.finfo(np.float32).min
FLOAT_MAX = np.finfo(np.float32).max


def sequence_hr_score(model, test, k=[10], exclude_preceding=True):
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


def sequence_ndcg_score(model, test, k=[10], exclude_preceding=True):
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
