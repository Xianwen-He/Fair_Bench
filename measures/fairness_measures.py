import numpy as np


### The violation functions are based on Jiang's work:
### https://github.com/google-research/google-research/tree/master/label_bias

#@title Helper Functions for DP
def get_error_and_violations(y_pred, y, protected_attributes):
  acc = 1 - np.mean(y_pred != y)
  violations = []
  for p in protected_attributes:
    protected_idxs = np.where(p > 0)
    violations.append(np.mean(y_pred>0) - np.mean(y_pred[protected_idxs]>0))
  pairwise_violations = []
  for i in range(len(protected_attributes)):
    for j in range(i+1, len(protected_attributes)):
      protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
      if len(protected_idxs[0]) == 0:
        continue
      pairwise_violations.append(np.mean(y_pred>0) - np.mean(y_pred[protected_idxs]>0))
  return acc, violations, pairwise_violations


#@title Helper Functions for Equal Opportunity
def get_error_and_violations_EqualOpportunity(y_pred, y, protected_attributes):
  acc =  1 - np.mean(y_pred != y)
  violations = []
  for p in protected_attributes:
    protected_idxs = np.where(np.logical_and(p > 0, y > 0))
    positive_idxs = np.where(y > 0)
    violations.append(np.mean(y_pred[positive_idxs]>0) - np.mean(y_pred[protected_idxs]>0))
  pairwise_violations = []
  for i in range(len(protected_attributes)):
    for j in range(i+1, len(protected_attributes)):
      protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
      if len(protected_idxs[0]) == 0:
        continue
      pairwise_violations.append(np.mean(y_pred>0) - np.mean(y_pred[protected_idxs]>0))
  return acc, violations, pairwise_violations


#@title Helper Functions for Equalized Odds
def get_error_and_violations_EqualizedOdds(y_pred, y, protected_attributes):
  acc = 1 - np.mean(y_pred != y)
  violations = []
  for p in protected_attributes:
    protected_idxs = np.where(np.logical_and(p > 0, y > 0))
    positive_idxs = np.where(y > 0)
    violations.append(np.mean(y_pred[positive_idxs]>0) - np.mean(y_pred[protected_idxs]>0))
    protected_idxs = np.where(np.logical_and(p > 0, y < 1))
    negative_idxs = np.where(y < 1)
    violations.append(np.mean(y_pred[negative_idxs]>0) - np.mean(y_pred[protected_idxs]>0))
  pairwise_violations = []
  for i in range(len(protected_attributes)):
    for j in range(i+1, len(protected_attributes)):
      protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
      if len(protected_idxs[0]) == 0:
        continue
      pairwise_violations.append(np.mean(y_pred>0) - np.mean(y_pred[protected_idxs]>0))
  return acc, violations, pairwise_violations


### printing functions
# This aims for the selection of the violation function based on the name of the metric.
VIOLATION_DICN = {
    'Demographic Parity': get_error_and_violations,
    'Equal Opportunity': get_error_and_violations_EqualOpportunity,
    'Equalized Odds': get_error_and_violations_EqualizedOdds,
    'DP': get_error_and_violations,
    'EO': get_error_and_violations_EqualOpportunity,
    'EOs': get_error_and_violations_EqualizedOdds
}

# print the given metric
def check_print_scores(y_pred, y_true, protected_val, metric, print_acc = False):
    violation_func = VIOLATION_DICN[metric]
    acc, violations, pairwise_violations = violation_func(y_pred, y_true, protected_val)
    
    print('=={}=='.format(metric))
    if print_acc:
       print('Acc:', acc)
    print("Violation", max(np.abs(violations)), " \t\t All violations", violations)
    if len(pairwise_violations) > 0:
      print("Intersect Violations", max(np.abs(pairwise_violations)), " \t All violations", pairwise_violations)
    
    return 

# print all three types of violation scores
def check_print_scores_all(y_pred, y_true, protected_val):
    check_print_scores(y_pred, y_true, protected_val, 'Demographic Parity', True)
    check_print_scores(y_pred, y_true, protected_val, 'Equal Opportunity')
    check_print_scores(y_pred, y_true, protected_val, 'Equalized Odds')
    return



### Measures for evaluate the performances of fairness-enhancing methods
# acc 
def accuracy(y_pred, y_true, protected_indicator):
  acc = 1 - np.mean(y_pred != y_true)

  protected_idxs = np.where(protected_indicator > 0)
  protected_acc = 1 - np.mean(y_pred[protected_idxs] != y_true[protected_idxs])

  unprotected_idxs = np.where(protected_indicator < 1)
  unprotected_acc = 1 - np.mean(y_pred[unprotected_idxs] != y_true[unprotected_idxs])

  return acc, protected_acc, unprotected_acc


# dp
def demographic_parity(y_pred, y_true, protected_indicator):
 
  protected_idxs = np.where(protected_indicator > 0)
  return np.mean(y_pred > 0) - np.mean(y_pred[protected_idxs] > 0)


# eo
def equal_opportunity(y_pred, y_true, protected_indicator):
  positive_idxs = np.where(y_true > 0)
  protected_idxs = np.where(np.logical_and(protected_indicator > 0, y_true > 0))
  
  return np.mean(y_pred[positive_idxs] > 0) - np.mean(y_pred[protected_idxs] > 0)

# eos
def equalized_odds(y_pred, y_true, protected_indicator):

  positive_violation = equal_opportunity(y_pred, y_true, protected_indicator)
  
  negative_idxs = np.where(y_true < 1)
  protected_idxs = np.where(np.logical_and(protected_indicator > 0, y_true < 1))
  negative_violation = np.mean(y_pred[negative_idxs] > 0) - np.mean(y_pred[protected_idxs] > 0)

  return positive_violation, negative_violation


# acc, proc_acc, dp, eo, eos(negative only)
def get_all_scores(y_pred, y_true, protected_indicator):
  acc, proc_acc, _ = accuracy(y_pred, y_true, protected_indicator)
  
  dp = demographic_parity(y_pred, y_true, protected_indicator)

  eo = equal_opportunity(y_pred, y_true, protected_indicator)

  _, eos_negative = equalized_odds(y_pred, y_true, protected_indicator)

  return acc, proc_acc, dp, eo, eos_negative
