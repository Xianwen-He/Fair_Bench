"""
This file processes the data set Adult and Compas with 'sex' serving as the protected attribute.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import os

"""

Load adult data
return: train_df, test_df, feature_names, label_name

"""
def get_adult_data(data_folder, shuffle_seed=0):
  # customize the following paths if necessary
  PATH_TO_ADULT_TRAIN_FILE = "adult.data"
  PATH_TO_ADULT_TEST_FILE = "adult.test"
  PATH_TO_ADULT_TRAIN_FILE = os.path.join(data_folder, PATH_TO_ADULT_TRAIN_FILE)
  PATH_TO_ADULT_TEST_FILE = os.path.join(data_folder, PATH_TO_ADULT_TEST_FILE)

  CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country'
  ]
  CONTINUOUS_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
  ]
  COLUMNS = [  # 'fnlwgt' won't be included in the training later on
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
  ]
  LABEL_COLUMN = 'label'

  # readin data
  train_file = PATH_TO_ADULT_TRAIN_FILE
  test_file = PATH_TO_ADULT_TEST_FILE
  train_df_raw = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
  test_df_raw = pd.read_csv(
      test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
  
  # binary label
  train_df_raw[LABEL_COLUMN] = (
      train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  test_df_raw[LABEL_COLUMN] = (
      test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  
  # Preprocessing Features
  pd.options.mode.chained_assignment = None  # default='warn'

  ### Functions for preprocessing categorical and continuous columns.
  # binary columns
  def binarize_categorical_columns(input_train_df,
                                   input_test_df,
                                   categorical_columns=[]):

    def fix_columns(input_train_df, input_test_df):
      test_df_missing_cols = set(input_train_df.columns) - set(
          input_test_df.columns)
      for c in test_df_missing_cols:
        input_test_df[c] = 0
      train_df_missing_cols = set(input_test_df.columns) - set(
          input_train_df.columns)
      for c in train_df_missing_cols:
        input_train_df[c] = 0
      input_train_df = input_train_df[input_test_df.columns]
      return input_train_df, input_test_df

    # Binarize categorical columns.
    binarized_train_df = pd.get_dummies(
        input_train_df, columns=categorical_columns)
    binarized_test_df = pd.get_dummies(
        input_test_df, columns=categorical_columns)
    # Make sure the train and test dataframes have the same binarized columns.
    fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                binarized_test_df)
    return fixed_train_df, fixed_test_df
  
  # continuous columns
  def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
    assert (num_quantiles is None or bins is None)
    if num_quantiles is not None:
      train_quantized, bins_quantized = pd.qcut(
          input_train_df[continuous_column_name],
          num_quantiles,
          retbins=True,
          labels=False)
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins_quantized, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins_quantized, labels=False)
    elif bins is not None:
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins, labels=False)


  # Filter out all columns except the ones specified.
  train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                          [LABEL_COLUMN]]
  test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                        [LABEL_COLUMN]]
  
  # Bucketize continuous columns.
  bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
  bucketize_continuous_column(
      train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
  bucketize_continuous_column(
      train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
  bucketize_continuous_column(
      train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
  bucketize_continuous_column(
      train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
  train_df, test_df = binarize_categorical_columns(
      train_df,
      test_df,
      categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
  feature_names = list(train_df.keys())
  feature_names.remove(LABEL_COLUMN)
  num_features = len(feature_names)

  train_df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
  test_df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
  return train_df, test_df, feature_names, LABEL_COLUMN


"""

Load COMPAS data
return: train_df, test_df, feature_names, label_name

"""
def get_compas_data(data_folder, sensitive='race', flip=True,
                    split_ratio=0.66, random_seed=0):
  """
  sensitive: sensitive indicator, default to be race
  flip: whether to flip the label column, default to be true
  split_ratio: ratio for the training set
  """
  # customize the path if necessary
  PATH_TO_COMPAS_DATA = "compas-scores-two-years.csv"
  data_path = os.path.join(data_folder, PATH_TO_COMPAS_DATA)

  df = pd.read_csv(data_path)
  FEATURES = [
      'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
      'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
      'two_year_recid'
  ]

  df = df[FEATURES]
  df = df[df.days_b_screening_arrest <= 30]
  df = df[df.days_b_screening_arrest >= -30]
  df = df[df.is_recid != -1]
  df = df[df.c_charge_degree != 'O']
  df = df[df.score_text != 'N/A']
  # only consider blacks and whites for this analysis when race is the sensitive attributes
  if sensitive == 'race':
    df = df[df["race"].isin(["African-American", "Caucasian"])]
  
  continuous_features = [
      'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
  ]
  continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
  categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_df, categorical_columns=[]):
    # Binarize categorical columns.
    binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
    return binarized_df

  def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
    input_df[continuous_column_name] = pd.cut(
        input_df[continuous_column_name], bins, labels=False)

  for c in continuous_to_categorical_features:
    b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
    if c == 'priors_count':
      b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
    bucketize_continuous_column(df, c, bins=b)

  df = binarize_categorical_columns(
      df,
      categorical_columns=categorical_features +
      continuous_to_categorical_features)

  to_fill = [
      u'decile_score_0', u'decile_score_1', u'decile_score_2',
      u'decile_score_3', u'decile_score_4', u'decile_score_5'
  ]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
  to_fill = [
      u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
      u'priors_count_3.0', u'priors_count_4.0'
  ]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

  if sensitive == 'race':
    features = [
      u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
      u'race_African-American', u'race_Caucasian',
      # When race is selected as the sensitive attributes, those will be removed.
      # u'race_Hispanic', u'race_Native American', u'race_Other',
      u'score_text_High', u'score_text_Low', u'score_text_Medium',
      u'sex_Female', u'sex_Male',
      u'age_0', u'age_1', u'age_2', u'age_3', u'age_4', u'age_5',
      u'decile_score_0', u'decile_score_1', u'decile_score_2',
      u'decile_score_3', u'decile_score_4', u'decile_score_5',
      u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
      u'priors_count_3.0', u'priors_count_4.0']
  else:
    features = [
      u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
      u'race_African-American', u'race_Caucasian',
      # When race is selected as the sensitive attributes, those will be removed.
      u'race_Hispanic', u'race_Native American', u'race_Other',
      u'score_text_High', u'score_text_Low', u'score_text_Medium',
      u'sex_Female', u'sex_Male',
      u'age_0', u'age_1', u'age_2', u'age_3', u'age_4', u'age_5',
      u'decile_score_0', u'decile_score_1', u'decile_score_2',
      u'decile_score_3', u'decile_score_4', u'decile_score_5',
      u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
      u'priors_count_3.0', u'priors_count_4.0']
  label = 'two_year_recid'
  df = df[features + [label]]  # get all the columns required for exepriments

  # flip the label column so that 1 refers to an positive outcome
  if flip: 
    removed_label = df.pop(label).to_numpy()
    label = 'two_year_norecid'
    df[label] = 1.0 - removed_label
  
  # split the dataset
  if split_ratio <= 0.5:
    split_ratio = 1-split_ratio  # the training set is kept to be larger than the testing set
  train_df, test_df = train_test_split(df, test_size=1-split_ratio, random_state=random_seed)
  
  return train_df, test_df, features, label


# Default data division for the fairness-enhancing method.
class WrapData():
    def __init__(self, base_data_folder='../data'):
        self.base_data_folder = base_data_folder
        self.adult_folder = os.path.join(self.base_data_folder, 'adult')
        self.compas_folder = os.path.join(self.base_data_folder, 'compas')
        return
    
    def wrap_data(self, dataset_name):
        # a quick way to load dataset from the given base data folder
        if dataset_name in ['Adult', 'adult', 'ADULT']:
            return self.wrap_Adult_data(self.adult_folder)
        elif dataset_name in ['COMPAS', 'compas', 'Compas']:
            return self.wrap_COMPAS_data(self.compas_folder)
        else:
            raise ValueError('Invalid data name.')
            
    def wrap_Adult_data(self, base_folder='../data/adult'):
        # adult
        PROTECTED_GROUPS = [ # here we only consider gender
        'gender_Female', 'gender_Male'
        ]
        train_df, test_df, feature_names, label_column = get_adult_data(base_folder)
        
        # wrap the data sets
        X_train_adult = np.array(train_df[feature_names])
        y_train_adult = np.array(train_df[label_column]).flatten()
        X_test_adult = np.array(test_df[feature_names])
        y_test_adult = np.array(test_df[label_column]).flatten()
        protected_train_adult = [np.array(train_df[g]).flatten() for g in PROTECTED_GROUPS]
        protected_test_adult = [np.array(test_df[g]).flatten() for g in PROTECTED_GROUPS]

        return X_train_adult, y_train_adult, X_test_adult, y_test_adult, protected_train_adult, protected_test_adult


    def wrap_COMPAS_data(self, base_folder='../data/compas'):
        # compas
        PROTECTED_GROUPS = [
        'race_African-American', 'race_Caucasian'
        ]
        train_df, test_df, feature_names, label_column = get_compas_data(base_folder)
        
        # wrap the data sets
        X_train_compas = np.array(train_df[feature_names])
        y_train_compas = np.array(train_df[label_column]).flatten()
        X_test_compas = np.array(test_df[feature_names])
        y_test_compas = np.array(test_df[label_column]).flatten()
        # the sensitive variables is stored as the dummy variables [protected_indicators, unprotected indicators]
        protected_train_compas = [np.array(train_df[g]).flatten() for g in PROTECTED_GROUPS]
        protected_test_compas = [np.array(test_df[g]).flatten() for g in PROTECTED_GROUPS]

        return X_train_compas, y_train_compas, X_test_compas, y_test_compas, protected_train_compas, protected_test_compas