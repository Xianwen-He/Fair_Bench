import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # SVM for classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import os
import sys

from thresholding_functions import prediction_correction, customized_predict
sys.path.insert(0, '../')
from data.process_data import WrapData
from model.mlp import BinaryMLP
from model.model_utils import train_BinaryMLP, prepare_dataloader, collect_predictions
from measures.fairness_measures import check_print_scores_all


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


def train_Linear_prob(X_train, y_train, X_test, y_test,
                      model_name='LogReg', max_iteration=1000):
    if model_name == 'LogReg':
        model = LogisticRegression(max_iter=max_iteration)
    elif model_name == 'SVM':
        model = make_pipeline(StandardScaler(),
                           SVC(kernel='linear', probability=True, max_iter=max_iteration, random_state=42))
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    return y_pred_train, y_prob_train, y_pred_test, y_prob_test


def train_MLP_prob(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
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

    return y_pred_train, y_prob_train, y_pred_test, y_prob_test




if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description="Customized arguments.")
    parser.add_argument('--model', type=str, choices=['LogReg', 'SVM', 'MLP'], default= "LogReg",
                      help="Model trained on the representations")
    parser.add_argument('--datasets', type=str, default='Adult,COMPAS',
                        help='Data sets to evaluate on')
    parser.add_argument('--save_file', type=str, help='file name for results savings')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--max_iteration', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    ### data preparation
    dataset_names = args.datasets.split(',') 
    datwrap = WrapData()
    datas = [datwrap.wrap_data(data_name) for data_name in dataset_names]
  

    ### Experiments
    # create the folder for result saving if necessary
    if args.save_file is not None:
        if not os.path.isdir(args.save_folder):
            os.mkdir(args.save_folder)

    # process every dataset
    for dataset_idx, dataset_name in enumerate(dataset_names):
      print("Processing {} =======>".format(dataset_name))
      X_train, y_train, X_test, y_test, protected_train, protected_test = datas[dataset_idx]
    
      # model training
      if args.model == 'LogReg':
          y_pred_train, y_prob_train, y_pred_test, y_prob_test = train_Linear_prob(X_train, y_train, X_test, y_test,
                                                 'LogReg', args.max_iteration)
      elif args.model == 'SVM':
          y_pred_train, y_prob_train, y_pred_test, y_prob_test = train_Linear_prob(X_train, y_train, X_test, y_test,
                                                 'SVM', args.max_iteration)
      elif args.model == 'MLP':
          y_pred_train, y_prob_train, y_pred_test, y_prob_test = train_MLP_prob(X_train, y_train, X_test, y_test)


      # check the prediction for baseline
      print('=====Baseline results=====')
      print('Training Scores')
      check_print_scores_all(y_pred_train, y_train, protected_train)
      print('Testing Scores')
      check_print_scores_all(y_pred_test, y_test, protected_test)
      print()


      # debias technqiues
      test_result_dicn = {}
      print("=====Thresholding Methods=====")
      # save the baseline testing results
      test_result_dicn['label'] = y_test
      test_result_dicn['protected_indicator'] = protected_test[0]
      test_result_dicn['y_pred'] = y_pred_test
      test_result_dicn['y_prob'] = y_prob_test
    
      for method in ['even', 'protected', 'unprotected']:
          print("Strategy:", method, "=>")
          thresholds, sacrifice = prediction_correction(y_train, y_pred_train, y_prob_train, protected_train[0], method)
          print('Threshold for the unprotected group: {:.4f}; Threshold for the protected group: {:.4f}'.format(thresholds[0], thresholds[1]))
      
          new_y_pred_train = customized_predict(y_prob_train, protected_train[0], thresholds)
          new_y_pred_test = customized_predict(y_prob_test, protected_test[0], thresholds)
          test_result_dicn['y_pred_{}'.format(method)] = new_y_pred_test
    
          print('Training Scores')
          check_print_scores_all(new_y_pred_train, y_train, protected_train)
          print('Testing Scores')
          check_print_scores_all(new_y_pred_test, y_test, protected_test)
          print()
  

      ### save the testing results
      test_result_df = pd.DataFrame(test_result_dicn)
      save_file = dataset_name + '_' + args.save_file + '_' + args.model + '.csv'
      save_file = os.path.join(args.save_folder, save_file)
      test_result_df.to_csv(save_file, index=False)
      print('Results saved for dataset {}.'.format(dataset_name))
      print()
    
    print('All set.')