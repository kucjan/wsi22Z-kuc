from solver import Solver
import numpy as np
import pandas as pd
from process_data import process_data, dataset_split
import matplotlib.pyplot as plt
from math import inf
from statistics import mean
from copy import deepcopy

class ID3Tree(Solver):
  
  def __init__(self, max_depth):
    self.max_depth = max_depth
    self.tree = {}
    self.classes = []
    self.label = ''
    
  def get_parameters(self):
    params = {
      'max_depth': self.max_depth,
      'tree': self.tree,
      'classes': self.classes,
      'label': self.label
    }
    
    return params
  
  def fit(self, X, y, label):
    train_data = pd.concat([X, y], axis=1)
    train_data_copy = deepcopy(train_data)
    self.label = label
    self.classes = train_data[label].unique()
    tree = deepcopy(self.tree)
    self.id3(tree, None, train_data_copy, current_depth=0)
    
    self.tree = tree
  
  def predict(self, X, tree_copy):
    predictions = []
    for _, row in X.iterrows():
      predictions.append(self.predict_row(row, tree_copy))
    return predictions
    
  def predict_row(self, row, tree_copy):
    if not isinstance(tree_copy, dict):
      return tree_copy
    else:
      attr_node = next(iter(tree_copy))
      attr_value = row[attr_node]
      if attr_value in tree_copy[attr_node]:
        return self.predict_row(row, tree_copy[attr_node][attr_value])
      else:
        return None
      
  def evaluate(self, X, y, tree_copy):
    trues = []
    falses = []
    predictions = self.predict(X, tree_copy)
    for i in range(len(predictions)):
      if predictions[i] == y.iloc[i]:
        trues.append(i)
      else:
        falses.append(i)
        
    return len(trues) / (len(trues) + len(falses)) # accuracy
  
  def calc_dataset_entropy(self, data_part):
    row_count = data_part.shape[0]
    dataset_entropy = 0
    
    for c in self.classes:
      class_count = data_part[data_part[self.label] == c].shape[0]
      class_prob = class_count / row_count
      class_entropy = - (class_prob) * np.log(class_prob)
      dataset_entropy += class_entropy
    
    return dataset_entropy
  
  def calc_attr_entropy(self, attr_value_data):
    """A method to calculate entropy for part of dataset, 
    where rows have same value of given attribute"""
    attr_value_count = attr_value_data.shape[0]
    entropy = 0
    
    for c in self.classes:
      class_count = attr_value_data[attr_value_data[self.label] == c].shape[0]
      class_entropy = 0
      if class_count > 0:
        class_prob = class_count / attr_value_count
        class_entropy = - (class_prob) * np.log(class_prob)
      entropy += class_entropy
    
    return entropy

  def calc_attr_info_gain(self, attr_name, data_part):
    attr_values = data_part[attr_name].unique()
    attr_row_count = data_part.shape[0]
    attr_info = 0.0
    
    for attr_value in attr_values:
      attr_value_data = data_part[data_part[attr_name] == attr_value]
      attr_value_count = attr_value_data.shape[0]
      attr_value_prob = attr_value_count / attr_row_count
      attr_value_entropy = self.calc_attr_entropy(attr_value_data)
      attr_info += attr_value_prob * attr_value_entropy
      
    return self.calc_dataset_entropy(data_part) - attr_info
  
  def find_best_attr(self, data_part):
    attributes = data_part.columns.drop(self.label)
    
    best_info_gain = -1
    
    for attr in attributes:
      attr_info_gain = self.calc_attr_info_gain(attr, data_part)
      if attr_info_gain > best_info_gain:
        best_info_gain = attr_info_gain
        best_attr = attr
        
    return best_attr
  
  def subtree(self, attr_name, data_part):
    # count occurances of each value of given attribute in data left
    attr_values_count = data_part[attr_name].value_counts(sort=False)
    
    subtree = {}
    
    for attr_value, value_count in attr_values_count.items():
      # get part of data where given attribute has given value
      attr_value_data = data_part[data_part[attr_name] == attr_value]
      
      is_leaf = False
      
      for c in self.classes:
        # count class occurances for given attribute
        # in records where given attribute has given value
        class_count = attr_value_data[attr_value_data[self.label] == c].shape[0]
        
        # value_count means occurances of given attr value in dataset
        # if statement below is True, we've found node with pure class
        if class_count == value_count:
          # save pure class node
          subtree[attr_value] = c
          
          # removing rows with given attribute value
          data_part = data_part[data_part[attr_name] != attr_value]
          
          is_leaf = True
      
      # if not pure class found
      if not is_leaf:
        subtree[attr_value] = '?'
        
    return subtree, data_part
    
  def id3(self, node, prev_attr_value, data_part, current_depth):
    if data_part.shape[0] != 0:
      print(node)
      # find most informative attribute
      best_attr = self.find_best_attr(data_part)
      
      # print(f'current depth= {current_depth}, max_depth= {self.max_depth}, best_attr: {best_attr}')
      
      # create subtree for given attribute in node
      attr_tree, data_part = self.subtree(best_attr, data_part)
      
      # check if node is not starting root of tree
      if prev_attr_value != None:
        node[prev_attr_value] = {}
        node[prev_attr_value][best_attr] = attr_tree
        new_node = node[prev_attr_value][best_attr]
      else:
        node[best_attr] = attr_tree
        new_node = node[best_attr]
        current_depth = 0

      for attr_value, class_label in new_node.items():
        # check if node is expandable
        if class_label == '?':
          attr_value_data = data_part[data_part[best_attr] == attr_value]
          if current_depth < self.max_depth:
            current_depth += 1
            self.id3(new_node, attr_value, attr_value_data, current_depth)
          else:
            # count most labels
            class_count_dict = attr_value_data[self.label].value_counts(sort=True)
            new_node[attr_value] = class_count_dict.keys()[0]

if __name__ == '__main__':
  
  # give an absolute path to file with dataset
  FILENAME = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab4/data/cardio_train.csv'
  
  non_discrete_attrs = {'age': [-inf, 7300, 10950, 14600, 18250, 20075, 21900, 23725, 25550, inf],
                      'weight': [-inf, 50, 70, 90, 110, 130, inf],
                      'height': [-inf, 120, 150, 170, 190, 205, inf],
                      'ap_hi': [-inf, 120, 130, 140, 160, 180, inf],
                      'ap_lo': [-inf, 80, 85, 90, 100, 110, inf]}
  
  label = 'cardio'

  dataset = process_data(FILENAME, non_discrete_attrs)
  
  # DEPTHS = [5, 8, 10, 14, 18, 22, 28, 35]
  DEPTHS = [5]
  SPLITS = 1
  
  save_params = []
  mean_accs_train = []
  mean_accs_valid = []
  
  for i in range(SPLITS):
    train_data_X, train_data_y, valid_data_X, valid_data_y, test_data_X, test_data_y = dataset_split(dataset, label)
    split_accs_train = []
    split_accs_valid = []
    for depth in DEPTHS:
      decision_tree = ID3Tree(max_depth=depth)
      
      decision_tree.fit(X=train_data_X, y=train_data_y, label=label)
      
      accuracy_train = decision_tree.evaluate(X=train_data_X, y=train_data_y, tree_copy=deepcopy(decision_tree.tree))
      split_accs_train.append(accuracy_train)
      print(f'Accuracy train: {accuracy_train}')
      
      accuracy_valid = decision_tree.evaluate(X=valid_data_X, y=valid_data_y, tree_copy=deepcopy(decision_tree.tree))
      split_accs_valid.append(accuracy_valid)
      print(f'Accuracy valid: {accuracy_valid}')
      
      save_params.append(decision_tree.get_parameters())
      
    mean_accs_train.append(mean(split_accs_train))
    mean_accs_valid.append(mean(split_accs_valid))
    print(f'Mean accs (train, valid): {mean_accs_train}, {mean_accs_valid}')
  
  plt.figure()
  plt.plot(DEPTHS, mean_accs_train, 'b', label='training data')
  plt.plot(DEPTHS, mean_accs_valid, 'r', label='validation data')
  plt.legend(loc='upper left')
  plt.xlabel('max depths')
  plt.ylabel('accuracy')
  # plt.xlim(DEPTHS[0], DEPTHS[-1])
  # plt.ylim(0, 1)
  plt.show()
  
  print(save_params)
