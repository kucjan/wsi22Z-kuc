from solver import Solver
import numpy as np
from process_data import process_data
import json
import pprint
from copy import deepcopy

class SolverID3(Solver):
  
  def __init__(self, dataset, label, max_depth):
    # dataset split: 60% x 20% x 20%
    self.train_data, self.valid_data, self.test_data = np.split(dataset.sample(frac=1),
                                                                [int(.6*len(dataset)),
                                                                 int(.8*len(dataset))])
    self.label = label
    self.max_depth = max_depth
    self.current_depth = 0
    self.class_list = dataset[label].unique()
    self.tree = {}
    
  def get_parameters(self):
    return super().get_parameters()
  
  def fit(self, X, y):
    return super().fit(X, y)
  
  def predict(self, X, tree):
    predictions = []
    for index, row in X.iterrows():
      print(f'row: {row}')
      tree_copy = deepcopy(tree)
      if not isinstance(tree_copy, dict):
        predictions.append(tree_copy)
      else:
        root_node = next(iter(tree_copy))
        print(root_node)
        attr_value = row[root_node]
        if attr_value in tree_copy[root_node]:
          return self.predict(row, tree_copy[root_node][attr_value])
        else:
          return None
    return predictions
      
  
  def calc_dataset_entropy(self):
    row_count = self.train_data.shape[0]
    dataset_entropy = 0
    
    for c in self.class_list:
      class_count = self.train_data[self.train_data[self.label] == c].shape[0]
      class_prob = class_count / row_count
      class_entropy = - (class_prob)*np.log2(class_prob)
      dataset_entropy += class_entropy
    
    return dataset_entropy
  
  def calc_attr_entropy(self, attr_value_data):
    attr_value_count = attr_value_data.shape[0]
    entropy = 0
    
    for c in self.class_list:
      class_count = attr_value_data[attr_value_data[self.label] == c].shape[0]
      class_entropy = 0
      if class_count > 0:
        class_prob = class_count / attr_value_count
        class_entropy = - (class_prob)*np.log2(class_prob)
      entropy += class_entropy
    
    return entropy

  def calc_attr_info_gain(self, attr_name):
    attr_values = self.train_data[attr_name].unique()
    attr_row_count = self.train_data.shape[0]
    attr_info = 0.0
    
    for attr_value in attr_values:
      attr_value_data = self.train_data[self.train_data[attr_name] == attr_value]
      attr_value_count = attr_value_data.shape[0]
      attr_value_prob = attr_value_count / attr_row_count
      attr_value_entropy = self.calc_attr_entropy(attr_value_data)
      attr_info += attr_value_prob * attr_value_entropy
      
    return self.calc_dataset_entropy() - attr_info
  
  def find_best_attr(self):
    attributes = self.train_data.columns.drop(self.label)
    
    best_info_gain = -1
    best_attr = None
    
    for attr in attributes:
      attr_info_gain = self.calc_attr_info_gain(attr)
      if attr_info_gain > best_info_gain:
        best_info_gain = attr_info_gain
        best_attr = attr
        
    return best_attr
  
  def create_attr_tree(self, attr_name):
    attr_values_count_dict = self.train_data[attr_name].value_counts(sort=False)
    
    tree = {}
    
    for attr_value, value_count in attr_values_count_dict.items():
      attr_value_data = self.train_data[self.train_data[attr_name] == attr_value]
      
      is_node = False
      
      for c in self.class_list:
        # count class occurances for given attribute
        class_count = attr_value_data[attr_value_data[self.label] == c].shape[0]
        
        # value_count means occurances of given attr value in dataset
        # if statement below is True, we've found node with pure class
        if class_count == value_count:
          # save pure class node
          tree[attr_value] = c
          
          # removing rows with given attribute value
          self.train_data = self.train_data[self.train_data[attr_name] != attr_value]
          
          is_node = True
      
      # if not pure class
      if not is_node:
        tree[attr_value] = '?'
        
    return tree
  
  def id3(self, root, prev_attr_value):
    if self.train_data.shape[0] != 0:
      best_attr = self.find_best_attr()
      attr_tree = self.create_attr_tree(best_attr)
      next_root = None
      
      # check if it is not starting root of tree
      if prev_attr_value != None:
        root[prev_attr_value] = {}
        root[prev_attr_value][best_attr] = attr_tree
        next_root = root[prev_attr_value][best_attr]
      else:
        root[best_attr] = attr_tree
        next_root = root[best_attr]
        
      self.current_depth += 1
        
      for node, branch in next_root.items():
        # check if branch is expandable
        if branch == '?' and self.current_depth <= self.max_depth:
          self.train_data = self.train_data[self.train_data[best_attr] == node]
          self.id3(next_root, node)

if __name__ == '__main__':
  
  FILENAME = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab4/data/cardio_train.csv'
  
  non_discrete_attrs = {'age': [9125, 12775, 15330, 17520, 20075, 21900, 23725, 25550],
                      'weight': [0, 20, 40, 60, 90, 120, 150, 200],
                      'height': [40, 80, 120, 150, 170, 190, 205, 250],
                      'ap_hi': [-500, 800, 2000, 5000, 8000, 11000, 14000, 17000],
                      'ap_lo': [-300, 600, 1400, 3300, 6000, 8200, 10000, 13000]}
  
  dataset = process_data(FILENAME, non_discrete_attrs)
  
  id3_solver = SolverID3(dataset, 'cardio', max_depth=10)
  
  id3_solver.id3(id3_solver.tree, None)
  
  print(id3_solver.tree)
  
  pprint.pprint(id3_solver.tree, indent=4)
  
  print(id3_solver.test_data.iloc[:2,:])
  
  preds = id3_solver.predict(id3_solver.test_data.iloc[:2,:], id3_solver.tree)
  
  print(preds)
