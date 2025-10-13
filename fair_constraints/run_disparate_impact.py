### Updated by @Xianwen Feb 4, 2025 
import os, sys
import numpy as np
import argparse
import pandas as pd
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

sys.path.insert(0, '../')
from data.process_data import get_adult_data, get_compas_data
from measures.fairness_measures import check_print_scores_all


DATASET ={
	'Adult':{
		'folder': '../data/adult',
		'SENSITIVE_ATTRS': ['gender_Female', 'gender_Male'],
		'INDICATOR': 'gender_Male',  # identifier for the priviledged group as required by this method
		'func': get_adult_data
    },
	'COMPAS':{
		'folder': '../data/compas',
		'SENSITIVE_ATTRS': ['race_African-American', 'race_Caucasian'],
		'INDICATOR': 'race_Caucasian',
		'func': get_compas_data
    }
}


def train_test_DisparateImpact(data_name, loss_function):

	""" Load the data """
	profile = DATASET[data_name]
	print('==============Process {}=============='.format(data_name))
	### load data
	train_df, test_df, feature_names, label_name = profile['func'](profile['folder'])
	SENSITIVE_ATTRS = profile['SENSITIVE_ATTRS']
	INDICATOR = profile['INDICATOR']
	
    ### prepare for the modeling
    # split x, y, and sensitive attrs
	# prepare x for the linear classifier
	# convert y into {-1, 1}
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.process_train_test_df(train_df, test_df, label_name,
																							   SENSITIVE_ATTRS, [INDICATOR],
																								intercept=True, as_negative=True)
	ut.compute_p_rule(x_control_train[INDICATOR], y_train)

	# sensitive information
	sensitive_attrs = [INDICATOR]
	sensitive_attrs_to_cov_thresh = {}
	gamma = None
	# constraints
	apply_fairness_constraints = None
	apply_accuracy_constraint = None
	sep_constraint = None

	# result dictionary
	test_result_dicn = {'label': y_test,
					    'protected': 1-x_control_test[INDICATOR]}
	protected_test = [1-x_control_test[INDICATOR], x_control_test[INDICATOR]]
	
	""" Function to train and test a model under desired constraints """
	def train_test_classifier():
		# train a model under disired constraints
		w = ut.train_model(x_train, y_train, x_control_train, loss_function,
		apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
		sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
		
		train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
		distances_boundary_test = (np.dot(x_test, w)).tolist()
		all_class_labels_assigned_test = np.sign(distances_boundary_test)
		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
		
		return w, p_rule, test_score


	""" 1. Classify the data while optimizing for accuracy """
	print()
	print("== Unconstrained (original) classifier ==")
	# all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	w_uncons, p_uncons, acc_uncons = train_test_classifier()
	print("p-rule: {}; test_score: {}".format(p_uncons, acc_uncons))
	
	# save results
	y_pred = ut.get_predicted_label(w_uncons, x_test)
	test_result_dicn['unconstrained_pred'] = y_pred
	check_print_scores_all(ut.to_nonnegative(y_pred), ut.to_nonnegative(y_test), protected_test)
	
	""" 2. Now classify such that we optimize for accuracy while achieving perfect fairness """
	apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
	apply_accuracy_constraint = 0
	sep_constraint = 0
	sensitive_attrs_to_cov_thresh = {INDICATOR: 0}
	print()
	print("== Classifier with fairness constraint ==")
	w_f_cons, p_f_cons, acc_f_cons  = train_test_classifier()
	print("p-rule: {}; test_score: {}".format(p_f_cons, acc_f_cons))
	
	# save results
	y_pred = ut.get_predicted_label(w_f_cons, x_test)
	test_result_dicn['pure_fair_pred'] = y_pred
	check_print_scores_all(ut.to_nonnegative(y_pred), ut.to_nonnegative(y_test), protected_test)


	""" 3. Classify such that we optimize for fairness subject to a certain loss in accuracy """
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
	sep_constraint = 0
	gamma = 0.5 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
	print("== Classifier with accuracy constraint ==")
	w_a_cons, p_a_cons, acc_a_cons = train_test_classifier()
	print("p-rule: {}; test_score: {}".format(p_a_cons, acc_a_cons))	
	
	# save results
	y_pred = ut.get_predicted_label(w_a_cons, x_test)
	test_result_dicn['part_fair_pred'] = y_pred
	check_print_scores_all(ut.to_nonnegative(y_pred), ut.to_nonnegative(y_test), protected_test)

	""" 
	4. Classify such that we optimize for fairness subject to a certain loss in accuracy 
	In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

	"""
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
	sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
	gamma = 1000.0
	print("== Classifier with accuracy constraint (no +ve misclassification) ==")
	w_a_cons_fine, p_a_cons_fine, acc_a_cons_fine  = train_test_classifier()
	print("p-rule: {}; test_score: {}".format(p_a_cons_fine, acc_a_cons_fine))	
	
	# save results
	y_pred = ut.get_predicted_label(w_a_cons_fine, x_test)
	test_result_dicn['fine_part_fair_pred'] = y_pred
	check_print_scores_all(ut.to_nonnegative(y_pred), ut.to_nonnegative(y_test), protected_test)

	
	return test_result_dicn




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Customized arguments.")
	parser.add_argument('--datasets', type=str, default= "Adult,COMPAS",
                      help="Comma-separated list for datasets")
	parser.add_argument('--model', type=str, choices=['LogReg', 'SVM'], default='LogReg')
	parser.add_argument('--save_folder', type=str, default='./results')
	parser.add_argument('--save_file', type=str)
	args = parser.parse_args()

	# set up the folder for result saving
	if args.save_file is not None:
		if not os.path.isdir(args.save_folder):
			os.mkdir(args.save_folder)
	
	# set up the loss function for the designated model
	if args.model == 'LogReg':
		loss_function = lf._logistic_loss
	elif args.model == 'SVM':
		loss_function = lf._hinge_loss
	else:
		raise ValueError('Invalid model name.')
	
	# training on the datasets
	data_lst = args.datasets.split(',') 
	print(data_lst)
	for data_name in data_lst:
		test_result_dicn = train_test_DisparateImpact(data_name, loss_function)
		
		if args.save_file is not None:
			base_save_file = os.path.join(args.save_folder, args.save_file+'_'+args.model+'_'+data_name+'.csv')
			test_result_df = pd.DataFrame(test_result_dicn)
			test_result_df.to_csv(base_save_file, index=False)
			print('Results saved.')
	
	print('All set.')
		