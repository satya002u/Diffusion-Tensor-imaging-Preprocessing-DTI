import pandas as pd
import math
import os
import re
import numpy as np
import math
import numpy as np
from numpy import *
import os 
import re



def dataset(dir, train_list, test_list):
    data_list = os.listdir(dir)
    # list = [data_list[0][:-8]]
    
    # count=1
    # count2=1
    # for n in range(1,len(data_list)):

    # 	data = data_list[n][:-8]

    # 	if data != list[count-1]:
    # 		count+=1
    # 		list.append(data)
    	

    def zscore(NC, train, test):
        mean = np.mean(NC)
        std = np.std(NC)
        return np.true_divide((train - mean), std), np.true_divide((test - mean), std)
    def Remove_symmtry(matrix):
        newMatrix = []
        row = matrix.shape[0]
        col = matrix.shape[1]
        # row 1 take 0 to 115, row 2 take 1 to 115 and so on.
        for n in range(0, row): 
            newMatrix = np.append(newMatrix, matrix[n][n:col], axis=0)
            # check for faulty matrix
        if newMatrix.shape[0] != 6786:
            print ("Dimension error sample is possibily corrupted")
        return newMatrix

    np.random.shuffle(train_list)
    np.random.shuffle(test_list)

    # list_train = list[:int(len(list)*frac_train)]
    list_train = train_list

    count_train = 0
    data_train =[]
    group_train = []
    normal_zscore_train = []
    file_name = []
    for i in range (len(list_train)):
        for file in data_list:
            if file[:-8]==list_train[count_train]:
    #             print(file)
                grp = file[:-13]
                file_name.append(file)
                path = os.path.join(dir, file)
    #             print(path)
                dataset = np.load(path)
                data = Remove_symmtry(dataset)
                if np.isnan(data).any() == True :
                    print (path)
                where_are_NaNs = isnan(data)
                data[where_are_NaNs] = 0
                data_train.append(data)
                group_train.append(grp)
                if grp=='Normal':
                    normal_zscore_train.append(data)
                

        count_train+=1
    # np.save('file_name.npy',file_name)
    # list_test = list[int(len(list)*frac_test):]
    list_test = test_list
    count_test = 0
    data_test =[]
    group_test = []
    for i in range (len(list_test)):
        for file in data_list:
            if file[:-8]==list_test[count_test]:
    #             print(file)
                grp = file[:-13]
                path = os.path.join(dir, file)
    #             print(path)
                dataset = np.load(path)
                data = Remove_symmtry(dataset)
                if np.isnan(data).any() == True :
                    print (path)
                where_are_NaNs = isnan(data)
                data[where_are_NaNs] = 0
                data_test.append(data)
                group_test.append(grp)
                

        count_test+=1
        
    train_data = np.array(data_train)
    test_data = np.array(data_test)
    group_train = np.array(group_train)
    group_test = np.array(group_test)
    train_normal_zscore = np.array(normal_zscore_train)
    # train_data, test_data = zscore(train_normal_zscore, train_data, test_data)
    
    return train_data, group_train, test_data, group_test
    # return train, group_train, test, group_test

   
def convert_to_one_hot(labels, dataset):
    one_hot_labels = []

    if dataset == 'cn_ad':
        for x in labels:
            t=0
            if x =='Normal': #EMCI          Normal
                t=np.array([1, 0], dtype=np.float32)
           
            elif x =='AD':         #LMCI             AD
                t=np.array([0, 1], dtype=np.float32)
    #        else:
    #            print (x)
            one_hot_labels.append(t)
    
    if dataset == 'mci_ad':
        for x in labels:
            t=0
            if x =='EMCI': #EMCI          Normal
                t=np.array([1, 0], dtype=np.float32)
            # elif x =='LMCI':         #LMCI             AD
            #     t=np.array([1, 0], dtype=np.float32)
            # elif x =='SMC':         #LMCI             AD
            #     t=np.array([1, 0], dtype=np.float32)
           
            elif x =='AD':         #LMCI             AD
                t=np.array([0, 1], dtype=np.float32)
    #        else:
    #            print (x)
            one_hot_labels.append(t)

    if dataset == 'cn_mci':
        for x in labels:
            t=0
            if x =='Normal': #EMCI          Normal
                t=np.array([1, 0], dtype=np.float32)
           
            elif x =='EMCI':         #LMCI             AD
                t=np.array([0, 1], dtype=np.float32)
            # elif x =='LMCI':         #LMCI             AD
            #     t=np.array([0, 1], dtype=np.float32)
            # elif x =='SMC':         #LMCI             AD
            #     t=np.array([0, 1], dtype=np.float32)
    #        else:
    #            print (x)
            one_hot_labels.append(t)
    if dataset == 'cn_lmci':
        for x in labels:
            t=0
            if x =='Normal': #EMCI          Normal
                t=np.array([1, 0], dtype=np.float32)
           
            # elif x =='EMCI':         #LMCI             AD
            #     t=np.array([0, 1], dtype=np.float32)
            elif x =='LMCI':         #LMCI             AD
                t=np.array([0, 1], dtype=np.float32)
            # elif x =='SMC':         #LMCI             AD
            #     t=np.array([0, 1], dtype=np.float32)
    #        else:
    #            print (x)
            one_hot_labels.append(t)

    one_hot_labels = np.asarray(one_hot_labels)
    return one_hot_labels


'''Function to randomise data and label'''
def randomize(a,b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
   
    return shuffled_a, shuffled_b

def data_generator(data_dir, class_subset, train_list, test_list ):
    x_train, y_train, x_test, y_test  = dataset(data_dir, train_list, test_list)   
    x_train, y_train = randomize(x_train, y_train)
    x_test, y_test = randomize(x_test, y_test)
    x_val, y_val = randomize(x_test, y_test)  
    train_grp = convert_to_one_hot(y_train, class_subset)
    val_grp = convert_to_one_hot(y_val, class_subset)
    test_grp = convert_to_one_hot(y_test, class_subset)
    
    return x_train, train_grp,  x_test,  test_grp

