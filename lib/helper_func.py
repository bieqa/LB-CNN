"""
Based on "helper_func.py" in [1] (https://bieqa.github.io/deeplearning.html)
for data splitting and performance saving and display

[1] Liu, C., Ji, H. and Qiu, A. Convolutional Neural Network on Semi-Regular 
Triangulated Meshes and its Application to Brain Image Data. arXiv preprint 
arXiv:1903.08828, 2019.

Apr. 02, 2020  Modified by Shih-Gu Huang, 
    slightly modifying save_to_csv() and print_classification_performance() 
    and removing unused funtions
"""


"""
Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

#import tensorflow as tf
import pandas as pd
#import sklearn
import numpy as np
import math
import random
#import os

                
"""Helper functions"""
# Define functions for initializing variables and standard layers
# For now, this seems superfluous, but in extending the code to many more 
# layers, this will keep our code more readable

def train_valid_data(input_data, input_target, valid_ratio):
    '''
    A helper function to split the data into training, validation and testing sets
    :param input_data: 4D tensor (samples x channels x height x width)
    :param input_target: 2D tensor for the target labels of the samples-
    :param valid_ratio: ratio of validation samples from full (or training) samples
    :param test_ratio: ratio of test samples from full samples
    :param test_set: if there is a need to split testing set
    :return: 4D tensors (data_train, data_valid, and/or data_test) 
             2D tensors (target_train, target_valid, or target_test)
    '''

    # determine the index of training, validation and testing sets
    # randomly sample a fixed ratio of control and patients to create testing set
    num_subj = input_target.shape[0]
    
    num_control = np.sum(input_target[:,0])
    num_patient = np.sum(input_target[:,1])
    control_index_all = [i for i,aa in enumerate(input_target[:,0]) if aa == 1]
#    control_valid_index = random.sample(control_index_all, int(math.floor(valid_ratio*num_control)))
    control_valid_index = control_index_all[0:int(math.floor(valid_ratio*num_control))] # special case for subjets with multiple scans arranged side-by-side
    
    patient_index_all = [i for i,aa in enumerate(input_target[:,1]) if aa == 1] 
#    patient_valid_index = random.sample(patient_index_all, int(math.floor(valid_ratio*num_patient)))
    patient_valid_index = patient_index_all[0:int(math.floor(valid_ratio*num_patient))] # special case for subjets with multiple scans arranged side-by-side

    # validation set
    data_valid_index = np.concatenate((control_valid_index, patient_valid_index), axis=0)
    data_valid = input_data[data_valid_index]
    target_valid = input_target[data_valid_index]
    
    # randomly sample a fixed ratio of control and patients to create training and validation sets
    data_train_index = np.delete(np.arange(num_subj), data_valid_index)    
    data_train = input_data[data_train_index]
    target_train = input_target[data_train_index]
    
#    print('Number of control: ', num_control)
#    print('Number of patient: ', num_patient)
                     
    return data_train, target_train, data_valid, target_valid # early prediction                   


def generate_train_batch(train_data, train_labels, batch_size, patient_ratio, random_batch_sampling_train=False):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''

    if random_batch_sampling_train is True:
        # Construct mini-batch by random sampling of subjects from whole dataset
        random_idx = random.sample(np.arange(0, train_labels.shape[0]), batch_size)       
        batch_data = train_data[random_idx, ...]
        batch_label = train_labels[random_idx]
    else:
        control_index_all = [i for i,aa in enumerate(train_labels) if aa == 0]
        control_train_index = random.sample(control_index_all, int(math.floor((1.0-patient_ratio)*batch_size)))
        patient_index_all = [i for i,aa in enumerate(train_labels) if aa == 1] 
        patient_train_index = random.sample(patient_index_all, int(math.ceil(patient_ratio*batch_size)))

        # training batch set
        data_train_index = np.concatenate((control_train_index, patient_train_index), axis=0)
        batch_data = train_data[data_train_index, ...]
        batch_label = train_labels[data_train_index]

    return batch_data, batch_label


def save_to_csv(idx, acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean, file_name, wholetime, epochtime, mode='w', header=False):
  df = {'aaidx':idx, 'accuracy':acc, 'fscore':fscore, 'sensitivity':sensitivity, 'specificity':specificity, 
                      'precision':precision, 'ppv':ppv, 'npv':npv, 'gmean':gmean, 'whole time':wholetime, 'epoch time': epochtime}
  df = pd.DataFrame(data=df, index=[0])
  df.to_csv(file_name, mode=mode, header=header)  
  


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def print_classification_performance(prevalence, loss, acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean):
  print( '----------------------------')
#  print( 'Classification performance for %s: ' % dataset)
  print( 'Validation prevalence = %.4f' % prevalence)
  print( 'Validation loss = %.4f' % loss)
  print( 'Validation top1 accuracy = %.4f' % acc)
  print( 'Validation top1 fscore = %.4f' % fscore)
  print( 'Validation top1 sensitivity = %.4f' % sensitivity)
  print( 'Validation top1 specificity = %.4f' % specificity)
  print( 'Validation top1 precision = %.4f' % precision)
  print( 'Validation top1 ppv = %.4f' % ppv)
  print( 'Validation top1 npv = %.4f' % npv)
  print( 'Validation top1 gmean = %.4f' % gmean)
  print( '----------------------------')
      