import argparse
import numpy as np
import os
import sys
import random
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # SVM for classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch.nn as nn
import torch.optim as optim
import pandas as pd

sys.path.insert(0, '../')
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



def load_saved_npz(file_name, value_name='X'):
    npz_file = np.load(file_name)
    return npz_file[value_name]


def process_reprs(repr_folder):
    # training
    repr_train = load_saved_npz(os.path.join(repr_folder, 'Z_train.npz'))
    A_train = load_saved_npz(os.path.join(repr_folder, 'A_train.npz'))
    Y_train = load_saved_npz(os.path.join(repr_folder, 'Y_train.npz'))
    # valid
    repr_valid = load_saved_npz(os.path.join(repr_folder, 'Z_valid.npz'))
    A_valid = load_saved_npz(os.path.join(repr_folder, 'A_valid.npz'))
    Y_valid = load_saved_npz(os.path.join(repr_folder, 'Y_valid.npz'))
    # testing
    repr_test = load_saved_npz(os.path.join(repr_folder, 'Z_test.npz'))
    A_test = load_saved_npz(os.path.join(repr_folder, 'A_test.npz'))
    Y_test = load_saved_npz(os.path.join(repr_folder, 'Y_test.npz'))

    # concatenate training and valid sets
    repr_train = np.concatenate((repr_train, repr_valid))
    # print(repr_train.shape)
    A_train  = np.concatenate((A_train, A_valid))
    Y_train = np.concatenate((Y_train, Y_valid))

    # Convert A into dummy variables, with the protected identifier in the first place.
    # In default, A=1 refers to the protected group
    protected_train = [np.array(A_train, dtype=float).flatten(), np.array(1-A_train, dtype=float).flatten()]
    protected_test = [np.array(A_test, dtype=float).flatten(),  np.array(1-A_test, dtype=float).flatten()]

    return repr_train, repr_test, np.array(Y_train, dtype=float).flatten(), np.array(Y_test, dtype=float).flatten(), protected_train, protected_test



def train_repr_Linear(X_train, y_train, X_test, y_test, model_name='LogReg', max_iteration=10000):
    if model_name == 'LogReg':
        model = LogisticRegression(max_iter=max_iteration)
    elif model_name == 'SVM':
        model = make_pipeline(StandardScaler(),
                           SVC(kernel='linear', probability=True, max_iter=max_iteration, random_state=42))
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return y_pred_train, y_pred_test


def train_repr_MLP(X_train, y_train, X_test, y_test):
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

   return y_pred_train, y_pred_test



if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description="Customized arguments.")
    parser.add_argument("--experiment_folder", type=str, default='./experiments/laftr_new/adult',
                       required=True, help="Folder containing the reprs.")
    parser.add_argument('--transfer_folder', type=str, default= "transfer",
                      help="Folder to save the transferring results")
    parser.add_argument('--model', type=str, choices=['LogReg', 'SVM', 'MLP'], default= "LogReg",
                      help="Model trained on the representations")
    parser.add_argument('--save', type=str, help='file name for results savings')
    args = parser.parse_args()
    print(args)

    # load dataset
    npz_folder = os.path.join(args.experiment_folder, 'npz')
    repr_train, repr_test, Y_train, Y_test, protected_train, protected_test = process_reprs(npz_folder)

    # model training
    if args.model == 'LogReg':
        y_pred_train, y_pred_test = train_repr_Linear(repr_train, Y_train, repr_test, Y_test,
                                                 'LogReg', 1000)
    elif args.model == 'SVM':
        y_pred_train, y_pred_test = train_repr_Linear(repr_train, Y_train, repr_test, Y_test,
                                                 'SVM', 10000)
    elif args.model == 'MLP':
        y_pred_train, y_pred_test = train_repr_MLP(repr_train, Y_train, repr_test, Y_test)


    # check the prediction
    print('Training Scores')
    check_print_scores_all(y_pred_train, Y_train, protected_train)
    print('Testing Scores')
    check_print_scores_all(y_pred_test, Y_test, protected_test)

    # save the results
    if args.save is not None:
        transfer_folder = os.path.join(args.experiment_folder, args.transfer_folder)
        if not os.path.isdir(transfer_folder):
            os.mkdir(transfer_folder)
        save_file = os.path.join(transfer_folder, args.save)

        train_file = save_file+'_train.csv'
        test_file = save_file+'_test.csv'
        train_result_df = pd.DataFrame({
          'label': Y_train,
          'protected': protected_train[0],
          'y_pred': y_pred_train,
        })
        train_result_df.to_csv(train_file, index=False)
        test_result_df = pd.DataFrame({
          'label': Y_test,
          'protected': protected_test[0],
          'y_pred': y_pred_test
        })
        test_result_df.to_csv(test_file, index=False)
        print('Results saved.')

    print('All set.')
