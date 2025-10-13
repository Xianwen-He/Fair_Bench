import numpy as np
import random
import pandas as pd
from debias_functions import *
from utils import *
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

sys.path.insert(0, '../')
from data.process_data import WrapData
from model.mlp import BinaryMLP
from model.model_utils import train_BinaryMLP, prepare_dataloader, collect_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# global variables
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)



def train_baseline_MLP(X_train, y_train, X_test, y_test, protected_train, protected_test,
                       save=None):
    
   # create data loader
   train_set, train_loader = prepare_dataloader(X_train, y_train)
   test_set, test_loader = prepare_dataloader(X_test, y_test)

   # initialize model
   input_size = X_train.shape[1]
   base_model = BinaryMLP(input_size).to(device)

   # criteria
   criterion = nn.BCEWithLogitsLoss(reduction='none')  # No sigmoid, per-sample weights
   optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    
   # training
   train_BinaryMLP(base_model, train_loader, criterion, optimizer, device, epochs=100)

   # get baseline preds
   y_prob_train, y_pred_train = collect_predictions(base_model, train_loader, device)
   y_prob_test, y_pred_test = collect_predictions(base_model, test_loader, device)

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
          'y_prob': y_prob_train,
          'y_pred': y_pred_train,
      })
      train_result_df.to_csv(train_file, index=False)
      test_result_df = pd.DataFrame({
          'label': y_test,
          'protected': protected_test[0],
          'y_prob': y_prob_test,
          'y_pred': y_pred_test
      })
      test_result_df.to_csv(test_file, index=False)
      print('Results saved.')

   print()
   print()


def train_debias_MLP(X_train, y_train, X_test, y_test, protected_train, protected_test,
                     metric, n_iters=10, save=None):
    
   metric_func = VIOLATION_DICN[metric]
   debias_func = DEBIAS_DICN[metric]
    
   test_set, test_loader = prepare_dataloader(X_test, y_test)

   if metric in ['EOs', 'Equalized Odds']:
      multipliers = np.zeros(len(protected_train) * 2)
   else:
      multipliers = np.zeros(len(protected_train))
   weights = np.array([1.] * X_train.shape[0])
   learning_rate = 1.
   # n_iters = 10
   for it in range(n_iters):
      # dataloader with given weights
      train_set, train_loader = prepare_dataloader(X_train, y_train, weights)
    
      # initialize model
      input_size = X_train.shape[1]
      model = BinaryMLP(input_size).to(device)
      # criteria
      criterion = nn.BCEWithLogitsLoss(reduction='none')  # No sigmoid, per-sample weights
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      # training
      train_BinaryMLP(model, train_loader, criterion, optimizer, device, epochs=100)

      # update weights
      weights = debias_func(y_train, protected_train, multipliers)
      
      # update multipliers by the output
      y_prob_train, y_pred_train = collect_predictions(model, train_loader, device)
      acc, violations, pairwise_violations = metric_func(y_pred_train, y_train, protected_train)
      multipliers += learning_rate * np.array(violations)
      print('multipliers:', multipliers)

      if (it + 1) % n_iters == 0:
        # print(multipliers)
        y_prob_test, y_pred_test = collect_predictions(model, test_loader, device)
        
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
          'y_prob': y_prob_train,
          'y_pred': y_pred_train,
      })
      train_result_df.to_csv(train_file, index=False)
      test_result_df = pd.DataFrame({
          'label': y_test,
          'protected': protected_test[0],
          'y_prob': y_prob_test,
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
   parser.add_argument('--datasets', type=str, default= "Adult,COMPAS",
                      help="Comma-separated list for datasets")
   parser.add_argument('--n_iters', type=int, default=10)
   parser.add_argument('--save_folder', type=str, default='./results')
   parser.add_argument('--save_file', type=str)
   args = parser.parse_args()


   print('=====> EXPERIMENTS WITH MLP <=====')
   print()
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
         train_baseline_MLP(X_train, y_train, X_test, y_test, protected_train, protected_test, base_file_name)
      else:
         train_debias_MLP(X_train, y_train, X_test, y_test, protected_train, protected_test, metric,
                          args.n_iters, base_file_name)

   print('All set.')

    