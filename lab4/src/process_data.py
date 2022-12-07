import pandas as pd
import numpy as np

def process_data(filename, non_discrete_attrs):
  dataset = pd.read_csv(filename, delimiter=';')
  dataset = dataset.drop('id', axis=1)
  for name, bins in non_discrete_attrs.items():
    dataset[name] = pd.cut(dataset[name], bins, right=False, labels=range(0, len(bins)-1))
    
  dataset, missing_values_columns = missing_values(dataset, True, True)
  
  print(missing_values_columns)
  
  return dataset

def missing_values(dataset, space_to_nan, drop_empty):
  if space_to_nan:
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
      
  if drop_empty:
    dataset = dataset.drop(dataset.index[dataset.isnull().all(axis = 1)]).reset_index(drop = True)
  
  missing_values = dataset.isnull().sum()

  missing_values_percent = 100 * dataset.isnull().sum() / len(dataset)
      
  missing_values_df = pd.concat([missing_values, missing_values_percent], axis=1)
  
  missing_values_columns = missing_values_df.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

  return dataset, missing_values_columns

def dataset_split(dataset, label):
  # dataset split: 60% x 20% x 20%
  train_data, valid_data, test_data = np.split(dataset.sample(frac=1),
                                                            [int(.6*len(dataset)),
                                                              int(.8*len(dataset))])

  train_data_X = train_data.drop(label, axis=1); train_data_y = train_data[label]
  valid_data_X = valid_data.drop(label, axis=1); valid_data_y = valid_data[label]
  test_data_X = test_data.drop(label, axis=1); test_data_y = test_data[label]
  
  return train_data_X, train_data_y, valid_data_X, valid_data_y, test_data_X, test_data_y

if __name__ == '__main__':
  
  filename = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab4/data/cardio_train.csv'
  
  non_discrete_attrs = {'age': [9125, 12775, 15330, 17520, 20075, 21900, 23725, 25550],
                        'weight': [0, 20, 40, 60, 90, 120, 150, 200],
                        'height': [40, 80, 120, 150, 170, 190, 205, 250],
                        'ap_hi': [-500, 800, 2000, 5000, 8000, 11000, 14000, 17000],
                        'ap_lo': [-300, 600, 1400, 3300, 6000, 8200, 10000, 13000]}
  
  new_data = process_data(filename, non_discrete_attrs)
  
  print(new_data)
  
  train_data, valid_data, test_data = np.split(new_data.sample(frac=1),
                                                                [int(.6*len(new_data)),
                                                                 int(.8*len(new_data))])
  
  print(train_data.shape, valid_data.shape, test_data.shape)
  
  print(new_data['cardio'].unique())
  print(new_data[new_data['age'] == 3])
  
  new_data, miss_vals = missing_values(new_data, True, True)
  
  print(new_data)
  print(miss_vals)
  