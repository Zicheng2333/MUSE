import numpy as np
from sklearn import metrics

def find_best_f1_threshold_binary(y_true, y_prob):
    """
    For binary classification: find the threshold (among unique scores)
    that yields the best F1 score. Return (best_threshold, best_f1).
    """
    thresholds = np.unique(y_prob)
    best_threshold = 0.5
    best_f1 = 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
    return best_threshold, best_f1

def metrics_binary(y_true, y_prob):
    """
    Compute binary metrics:
      1) AUROC
      2) AUPRC
      3) Precision
      4) Recall
      5) F1
    Using the threshold that maximizes F1.
    """
    # Find the threshold that yields the best F1
    best_thr, best_f1 = find_best_f1_threshold_binary(y_true, y_prob)
    y_pred = (y_prob >= best_thr).astype(int)

    precision = metrics.precision_score(y_true, y_pred)
    recall    = metrics.recall_score(y_true, y_pred)
    f1        = best_f1

    # AUROC
    auroc = metrics.roc_auc_score(y_true, y_prob)
    # AUPRC
    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    auprc = metrics.auc(rec_curve, prec_curve)

    return {
        "auroc":      auroc,
        "auprc":      auprc,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
    }

# -------------------------------------------------------------------------
# Multi-label
# -------------------------------------------------------------------------
def find_best_macro_f1_threshold_multilabel(y_true, y_prob):
    """
    For multi-label classification:
    - y_true: shape [N, C]  (0/1)
    - y_prob: shape [N, C]  (continuous in [0, 1])
    We find a *single* threshold that maximizes the *macro-F1* across all labels.
    Return (best_threshold, best_f1).
    """
    # Flatten out all predicted probabilities to find unique thresholds
    all_scores = np.unique(y_prob)
    best_thr = 0.5
    best_f1 = 0.0

    for thr in all_scores:
        y_pred = (y_prob >= thr).astype(int)
        # macro-F1
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1

def metrics_multilabel(y_true, y_prob):
    """
    Compute multi-label metrics (macro-averaged):
      1) AUROC (macro)
      2) AUPRC (macro)
      3) Precision (macro) at best-F1 threshold
      4) Recall (macro) at best-F1 threshold
      5) F1 (macro) at best-F1 threshold
    """
    # 1) Find the threshold that yields the best *macro-F1*
    best_thr, best_f1 = find_best_macro_f1_threshold_multilabel(y_true, y_prob)
    y_pred = (y_prob >= best_thr).astype(int)

    # 2) Compute macro precision, recall, F1 at that threshold
    precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)

    # 3) Compute macro AUROC
    #    Note: For multi-label tasks, `roc_auc_score` with average='macro'
    #    sums up each label's AUROC, then divides by C.
    auroc_macro = metrics.roc_auc_score(y_true, y_prob, average='macro')

    # 4) Compute macro AUPRC
    #    `average_precision_score(y_true, y_prob, average='macro')`
    #    for multi-label data is also well-defined in scikit-learn >= 0.19.
    auprc_macro = metrics.average_precision_score(y_true, y_prob, average='macro')

    return {
        "auroc":      auroc_macro,
        "auprc":      auprc_macro,
        "precision":  precision,
        "recall":     recall,
        "f1":         best_f1,
    }

# -------------------------------------------------------------------------
# Bootstrapping helpers
# -------------------------------------------------------------------------
def select_sample_with_replacement(y_true, y_prob, num_samples):
    """
    Randomly select 'num_samples' items from [0..len(y_true)-1] with replacement.
    Returns (y_true_sample, y_prob_sample).
    """
    N = y_true.shape[0]
    idx = np.random.choice(N, num_samples, replace=True)
    return y_true[idx], y_prob[idx]

def bootstrap_evaluation(y_true, y_prob, metric_fn, num_iterations=1000, num_samples=None):
    """
    Perform bootstrapping: sample 'num_samples' times with replacement from (y_true, y_prob).
    Then compute the metric(s) via 'metric_fn'. Repeat 'num_iterations' times.
    Return mean & std of each metric across all iterations.

    metric_fn should return a dict: {"auroc":..., "auprc":..., "precision":..., "recall":..., "f1":...}
    """
    if num_samples is None:
        num_samples = y_true.shape[0]

    # We'll collect each iteration's results in a dict of lists
    all_results = {}

    for _ in range(num_iterations):
        # Sample with replacement
        y_true_samp, y_prob_samp = select_sample_with_replacement(y_true, y_prob, num_samples)

        # We might need to check for trivial cases where it doesn't contain 0 or 1
        # in binary or doesn't contain any positives in multi-label, etc.
        # For simplicity, we just do a quick check if needed:
        # (Binary example: require at least one positive or negative)
        # If your multi-label data is large, it's unlikely to be an issue.
        iteration_result = metric_fn(y_true_samp, y_prob_samp)

        for k, v in iteration_result.items():
            all_results.setdefault(k, []).append(v)

    # Now compute mean and std for each metric
    final = {}
    for k, v_list in all_results.items():
        v_array = np.array(v_list, dtype=float)
        final[k] = v_array.mean()
        final[k + "_std"] = v_array.std()
    return final

# -------------------------------------------------------------------------
# Public functions you can call
# -------------------------------------------------------------------------
def get_metrics_binary(y_true, y_score, bootstrap=False, num_iterations=1000, num_samples=None):
    """
    For binary classification.
    If bootstrap=True, repeat 'num_iterations' resamples with replacement
    and return mean & std for each metric. Otherwise, return single-run metrics.
    """
    if not bootstrap:
        return metrics_binary(y_true, y_score)
    else:
        return bootstrap_evaluation(
            y_true, y_score, metrics_binary,
            num_iterations=num_iterations,
            num_samples=num_samples
        )

def get_metrics_multilabel(y_true, y_score, bootstrap=False, num_iterations=1000, num_samples=None):
    """
    For multi-label classification.
    If bootstrap=True, repeat 'num_iterations' resamples with replacement
    and return mean & std for each metric. Otherwise, return single-run metrics.
    """
    if not bootstrap:
        return metrics_multilabel(y_true, y_score)
    else:
        return bootstrap_evaluation(
            y_true, y_score, metrics_multilabel,
            num_iterations=num_iterations,
            num_samples=num_samples
        )