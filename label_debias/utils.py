import numpy as np
import sys
from debias_functions import *

VIOLATION_DICN = {
    'DP': get_error_and_violations,
    'EO': get_error_and_violations_EqualOpportunity,
    'EOs': get_error_and_violations_EqualizedOdds,
    'Demographic Parity': get_error_and_violations,
    'Equal Opportunity': get_error_and_violations_EqualOpportunity,
    'Equalized Odds': get_error_and_violations_EqualizedOdds
}


DEBIAS_DICN = {
    'Demographic Parity': debias_weights,
    'Equal Opportunity': debias_weights_EqualOpportunity,
    'Equalized Odds': debias_weights_EqualizedOdds,
    'DP': debias_weights,
    'EO': debias_weights_EqualOpportunity,
    'EOs': debias_weights_EqualizedOdds
}


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

def check_print_scores_all(y_pred, y_true, protected_val):
    check_print_scores(y_pred, y_true, protected_val, 'Demographic Parity', True)
    check_print_scores(y_pred, y_true, protected_val, 'Equal Opportunity')
    check_print_scores(y_pred, y_true, protected_val, 'Equalized Odds')
    return