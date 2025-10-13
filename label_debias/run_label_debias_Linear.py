import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC  # SVM for classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from debias_functions import *
from utils import *
import sys
import os
import argparse

sys.path.insert(0, '../')
from data.process_data import WrapData


# global variables
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)


def train_baseline_Linear(X_train, y_train, X_test, y_test, protected_train, protected_test,
                          model_name='LogReg', max_iteration=1000,
                          save=None):
   if model_name == 'LogReg':
      model = LogisticRegression(max_iter=max_iteration)
   elif model_name == 'SVM':
      model = make_pipeline(StandardScaler(),
                              SVC(kernel='linear', probability=True, max_iter=max_iteration, random_state=42))
    
   model.fit(X_train, y_train)
   y_pred_train = model.predict(X_train)
   y_pred_test = model.predict(X_test)


   # training and testing scores
   print("Training Scores")
   check_print_scores_all(y_pred_train, y_train, protected_train)
   print('Testing Scores')
   check_print_scores_all(y_pred_test, y_test, protected_test)
 
   # save results
   if save is not None:
      train_file = save+'_train.csv'
      test_file = save+'_test.csv'
      train_result_df = pd.DataFrame({
          'label': y_train,
          'protected': protected_train[0],
          'y_pred': y_pred_train,
      })
      train_result_df.to_csv(train_file, index=False)
      test_result_df = pd.DataFrame({
          'label': y_test,
          'protected': protected_test[0],
          'y_pred': y_pred_test
      })
      test_result_df.to_csv(test_file, index=False)
      print('Results saved.')
   print()
   print()


def train_debias_Linear(X_train, y_train, X_test, y_test, protected_train, protected_test,
                        metric, n_iters=100,
                        model_name='LogReg', max_iteration=1000,
                        save=None):
    
   metric_func = VIOLATION_DICN[metric]
   debias_func = DEBIAS_DICN[metric]

   if metric in ['EOs', 'Equalized Odds']:
      multipliers = np.zeros(len(protected_train) * 2)
   else:
      multipliers = np.zeros(len(protected_train))
   weights = np.array([1.] * X_train.shape[0])
   learning_rate = 1.
   # n_iters = 10
   for it in range(n_iters):
      if model_name == 'LogReg':
         model = LogisticRegression(max_iter=max_iteration)
         model.fit(X_train, y_train, weights)
      elif model_name == 'SVM':
         model = make_pipeline(StandardScaler(),
                           SVC(kernel='linear', probability=True, max_iter=max_iteration, random_state=42))
         model.fit(X_train, y_train, svc__sample_weight=weights)

      # prediction
      y_pred_train = model.predict(X_train)

      # update weights
      weights = debias_func(y_train, protected_train, multipliers)

      # update multipliers by the output
      acc, violations, pairwise_violations = metric_func(y_pred_train, y_train, protected_train)
      multipliers += learning_rate * np.array(violations)
      print('multipliers:', multipliers)

      if (it + 1) % n_iters == 0:
        # print(multipliers)
        y_pred_test = model.predict(X_test)
        
        print('Training Scores')
        check_print_scores_all(y_pred_train, y_train, protected_train)
        print('Testing Scores')
        check_print_scores_all(y_pred_test, y_test, protected_test)
        
   # save results
   if save is not None:
      train_file = save+'_train.csv'
      test_file = save+'_test.csv'
      train_result_df = pd.DataFrame({
          'label': y_train,
          'protected': protected_train[0],
          'y_pred': y_pred_train,
      })
      train_result_df.to_csv(train_file, index=False)
      test_result_df = pd.DataFrame({
          'label': y_test,
          'protected': protected_test[0],
          'y_pred': y_pred_test
      })
      test_result_df.to_csv(test_file, index=False)
      print('Results saved.')
   print()
   print()





if __name__ == '__main__':
   set_seed(42)
   parser = argparse.ArgumentParser(description="Customized arguments.")
   parser.add_argument("--metric", type=str, choices=["base", "DP", "EO", "EOs"], default='base',
                       required=True, help="Metric to used for debias.")
   parser.add_argument('--model', type=str, choices=['LogReg', 'SVM'], default='LogReg',
                       help='Linear model to use.')
   parser.add_argument('--max_iteration', type=int, default=1000)
   parser.add_argument('--n_iters', type=int, default=10)
   parser.add_argument('--datasets', type=str, default= "Adult,COMPAS",
                      help="Comma-separated list for datasets")
   parser.add_argument('--save_folder', type=str, default='./results')
   parser.add_argument('--save_file', type=str)

   args = parser.parse_args()


   print('=====> EXPERIMENTS WITH {} <====='.format(args.model))
   print(args)
   print()
  

   ### data preparation
   dataset_names = args.datasets.split(',') 
   datwrap = WrapData()
   datas = [datwrap.wrap_data(data_name) for data_name in dataset_names]
  

   ### Experiments
   # create the folder for result saving if necessary
   if args.save_file is not None:
      if not os.path.isdir(args.save_folder):
         os.mkdir(args.save_folder)

   metric = args.metric
   print("=====Experiments on metric {}=====".format(metric))
   for dataset_idx, dataset_name in enumerate(dataset_names):
      print("Processing ", dataset_name)
      X_train, y_train, X_test, y_test, protected_train, protected_test = datas[dataset_idx]

      if args.save_file is not None:
         base_file_name = os.path.join(args.save_folder, args.save_file+'_'+dataset_name)
      else:
         base_file_name = None

      if metric == 'base':
         train_baseline_Linear(X_train, y_train, X_test, y_test, protected_train, protected_test,
                               model_name=args.model, max_iteration=args.max_iteration,
                               save=base_file_name)
      else:
         train_debias_Linear(X_train, y_train, X_test, y_test, protected_train, protected_test,
                             metric, n_iters=args.n_iters,
                             model_name=args.model, max_iteration=args.max_iteration,
                             save=base_file_name)

   print('All set.')

    