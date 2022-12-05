import pandas as pd
import numpy as np
from random import shuffle

def process_data(filename, non_discrete_attrs):
  dataset = pd.read_csv(filename, delimiter=';')
  dataset = dataset.drop('id', axis=1)
  for name, bins in non_discrete_attrs.items():
    dataset[name] = pd.cut(dataset[name], bins, right=False, labels=range(0, len(bins)-1))
  
  return dataset

if __name__ == '__main__':
  
  filename = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab4/data/cardio_train.csv'
  
  non_discrete_attrs = {'age': [9125, 12775, 15330, 17520, 20075, 21900, 23725, 25550],
                        'weight': [0, 20, 40, 60, 90, 120, 150, 200],
                        'height': [40, 80, 120, 150, 170, 190, 205, 250],
                        'ap_hi': [-500, 800, 2000, 5000, 8000, 11000, 14000, 17000],
                        'ap_lo': [-300, 600, 1400, 3300, 6000, 8200, 10000, 13000]}
  
  new_data = process_data(filename, non_discrete_attrs)
  new_data = new_data.drop('id', axis=1)
  
  print(new_data)
  
  train_data, valid_data, test_data = np.split(new_data.sample(frac=1),
                                                                [int(.6*len(new_data)),
                                                                 int(.8*len(new_data))])
  
  print(train_data.shape, valid_data.shape, test_data.shape)
  
  print(new_data['cardio'].unique())
  print(new_data[new_data['age'] == 3])
  