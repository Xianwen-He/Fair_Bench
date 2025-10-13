import numpy as np



# Demographic Parity
def prediction_correction(original_labels, predicted_labels, 
                          predicted_probs, protected_indicators,
                          method='protected', bound=0.1):
  """
  This function takes the original prediction and returns customized threshold which would correct bias.
  So far, it can only handle the case where A is one-dimensional.

  Input:
    original_labels: y, predicted_labels: y_hat, predicted_probes: P(y_hat=1 | x, a)
    protected_indicators: an 1-D array that indicates whether each instance belongs to the protected group
    bound: upper bound for the sacrifice of accuracy
    method: ['even', 'protected', 'unprotected'], indicator for which threshold to alter

  Output: (threhold_for_unprotected, threshold_for_protected), sacrifice_for_acc
  """
  predicted_labels = np.array(predicted_labels, dtype=float).flatten()
  predicted_probs = np.array(predicted_probs, dtype=float).flatten()
  protected_indicators = np.array(protected_indicators).flatten()


  protected_idxs = protected_indicators > 0
  unprotected_ratio = np.mean(predicted_labels[~protected_idxs] > 0)
  protected_ratio = np.mean(predicted_labels[protected_idxs] > 0)
  
  # We only protect individuals belonging to the protected group here
  # That is, we alter the threshold only when P(Y_hat=1|A=1) < P(Y_hat|A=0)
  gap_ratio = min(max(unprotected_ratio-protected_ratio, 0.0), bound)
  print('Gap between groups is {:.4f} %. Tuning ratio is {:.4f} %'.format((unprotected_ratio-protected_ratio)*100, gap_ratio*100))
  # Nothing needs to be done if P(Y_hat=1|A=1) >= P(Y_hat|A=0)
  if not gap_ratio > 0:
    print('Thresholds are maintained for both groups. Nothing to be done')
    return (0.5, 0.5), 0.0
  
  if method == 'even':
    down_ratio = -0.5*gap_ratio
    up_ratio = gap_ratio*0.5
  elif method == 'protected':
    down_ratio = 0.0
    up_ratio = gap_ratio
  elif method == 'unprotected':
    down_ratio = -1.0*gap_ratio
    up_ratio = 0.0
  else:
    raise ValueError('Invalid tuning plan!')
  
  # sacrifice acc for fairness
  new_protected_ratio = up_ratio + protected_ratio
  new_unprotected_ratio = down_ratio + unprotected_ratio
  new_protected_threshold = np.quantile(predicted_probs[protected_idxs], 1.0-new_protected_ratio)
  new_unprotected_threshold = np.quantile(predicted_probs[~protected_idxs], 1.0-new_unprotected_ratio)
  
  return (new_unprotected_threshold, new_protected_threshold), gap_ratio




def customized_predict(predicted_probs, protected_indicators, thresholds):
  """
  prediction with given customized thresholds.
  Input:
  predicted_probs: an 1-D array-like object contraining P(y_hat=1 | x, a)
  protected_indicators: an 1-D array that indicates whether each instance belongs to the protected group
  thresholds: (threshold_for_unprotected_group, threshold_for_protected_group)
  
  Output: an 1-D array contraining customized prediction labels
  """
  predicted_probs = np.array(predicted_probs).flatten()
  protected_indicators = np.array(protected_indicators).flatten()

  customized_labels = np.ones(len(protected_indicators))

  protected_idxs = protected_indicators > 0

  customized_labels[protected_idxs] = np.array(predicted_probs[protected_idxs] >= thresholds[1], dtype=float)
  customized_labels[~protected_idxs] = np.array(predicted_probs[~protected_idxs] >= thresholds[0], dtype=float)
  # print("Gap:", np.mean(customized_labels[~protected_idxs]) - np.mean(customized_labels[protected_idxs]))
  
  return customized_labels
