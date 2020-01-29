import numpy as np
import sklearn.metrics


def hierarchical_macro_averaged_recall(pred_g, true_g,
                                       pred_v, true_v,
                                       pred_c, true_c):
    score_g = sklearn.metrics.recall_score(
        true_g, pred_g, average='macro'
    )
    score_v = sklearn.metrics.recall_score(
        true_v, pred_v, average='macro'
    )
    score_c = sklearn.metrics.recall_score(
        true_c, pred_c, average='macro'
    )
    scores = [score_g, score_v, score_c]
    return np.average(scores, weights=[2, 1, 1])