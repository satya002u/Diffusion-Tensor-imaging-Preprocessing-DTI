from __future__ import print_function
from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
import numpy as np
import keras
from sklearn import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras import backend as K
from keras.engine.topology import Layer
from deep_contribution_score import *
# from data_gen import data_generator
import os
from rfe_sensitivity import *
from numpy.random import seed
from keras import optimizers
from tensorflow import set_random_seed
from hadamard_layer import Hadamard
from fix_layer import Fixed_Layer 

from data_gen_new import data_generator


def generate_prop_number_of_neurons(input_multiplier, n1, n2):
    
    # new_input_size = int(round(current_input_size * input_multiplier, -1 * (len(str(round(current_input_size * input_multiplier))) - 5)))
    
    # if i == 1:
    #     new_input_size = 34716
    
    num_hidden_neurons_1 = int(round(n1*input_multiplier))
    num_hidden_neurons_2 = int(round(n2*input_multiplier))
    # num_hidden_neurons_3 = int(round(n3*input_multiplier))

    # print(new_input_size, num_hidden_neurons_1, num_hidden_neurons_2, num_hidden_neurons_3)

    return (num_hidden_neurons_1, num_hidden_neurons_2)

def generate_compunded_neurons(multipler, number_of_values):
    neuron_list = []
    start = 1
    count = 0
    while(number_of_values):
        neuron_list.append(start * (multipler ** count))
        count += 1
        number_of_values -= 1
    
    return neuron_list

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    
    return shuffled_a, shuffled_b

mask_saving = './results/masked_data/'
import os
import shutil
if not os.path.exists(mask_saving):
    os.makedirs(mask_saving)
else:
    shutil.rmtree(mask_saving)           #removes all the subdirectories!
    os.makedirs(mask_saving)


# classification_type = 'cn-ad' 
# class_subset = classification_type 
# classification_list=['cn-ad', 'cn-emci', 'emci-ad']
mode = 'average'
# dataset = class_subset


# from utils import plot, plt
# %matplotlib inline
gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  
data_dir  = '../data/dataset/'
mode = 'average'
# data_dir = './data/'
# class_type = 'nc_ad'
batch_size = 10
num_classes = 2
epochs = 100

contr_score_type = ['deeplift', 'grad_input', 'saliency', 'intgrad', 'elrp', 'occlusion']
# contr_score_type = ['grad_input', 'intgrad', 'elrp', 'saliency', 'occlusion']

classification_list=['cn_ad', 'cn_mci', 'mci_ad']
# classification_list=['cn_ad']

for contr_score in contr_score_type:

    for classification_type in classification_list:


        if classification_type == 'cn-ad':
            target_names = ['Normal', 'AD']
            class_type = 'nc_ad'
        elif classification_type == 'cn-mci':
            target_names = ['Normal', 'EMCI']
            class_type = 'nc_mci'
        elif classification_type == 'cn-lmci':
            target_names = ['Normal', 'LMCI']
        elif classification_type == 'mci-ad':
            target_names = ['EMCI', 'AD']
            class_type = 'mci_ad'

        

        important_features ='./results_'+contr_score+'/important_features_'+classification_type+'/'
        if not os.path.exists(important_features):
            os.makedirs(important_features)
        else:
            shutil.rmtree(important_features)           #removes all the subdirectories!
            os.makedirs(important_features)

        training_logs = './results_'+contr_score+'/training_logs_'+classification_type+'/'
        if not os.path.exists(training_logs):
            os.makedirs(training_logs)
        else:
            shutil.rmtree(training_logs)           #removes all the subdirectories!
            os.makedirs(training_logs)

        init_num_hidden_neuron_1 = 250
        init_num_hidden_neuron_2 = 150
        init_num_hidden_neuron_3 = 100
        mask = np.ones(6786)
        thresholds = generate_compunded_neurons(0.9, 46)

        if classification_type == 'cn_ad':
            batch_size = 6
            learning_rate = 0.01
            decay = 0.0001
            dropout = 0.0
            init_num_hidden_neurons_1, init_num_hidden_neurons_2, init_num_hidden_neurons_3 = 1450, 750, 450

        elif classification_type == 'cn_mci':
            batch_size = 4
            learning_rate = 0.02
            decay = 0.001
            dropout = 0.1
            init_num_hidden_neurons_1, init_num_hidden_neurons_2, init_num_hidden_neurons_3 = 1500, 800, 620

        elif classification_type == 'mci_ad':
            batch_size = 5
            learning_rate = 0.012
            decay = 0.001
            dropout = 0.1
            init_num_hidden_neurons_1, init_num_hidden_neurons_2, init_num_hidden_neurons_3 = 1250, 650, 520

        models = []
        # _x_train = np.load(data_dir  + class_type + '/x_train.npy')
        # _y_train = np.load(data_dir+ class_type + '/y_train.npy')
        # _x_test = np.load(data_dir+ class_type + '/x_test.npy')
        # _y_test = np.load(data_dir+ class_type + '/y_test.npy')
        # for x in range(5):
        with open(training_logs + 'training_logs_train_acc_' + classification_type + '.csv', 'a') as out_stream:
                    # out_stream.write(str(i)  + ', ' + str(sensitivity) + ', ' + str(specificity) + ', ' + str(f1) + ', ' + str(valacc) + ', ' + str(test_acc) + ', ' + str(np.shape(x_train)) + '\n')
            out_stream.write(','+ '\n')
        with open(training_logs + 'training_logs_test_acc_' + classification_type + '.csv', 'a') as out_stream:
                    # out_stream.write(str(i)  + ', ' + str(sensitivity) + ', ' + str(specificity) + ', ' + str(f1) + ', ' + str(valacc) + ', ' + str(test_acc) + ', ' + str(np.shape(x_train)) + '\n')
            out_stream.write( ','+'\n')

        for ix, i in enumerate(thresholds):
            # (num_hidden_neurons_1, num_hidden_neurons_2) = generate_prop_number_of_neurons(i, init_num_hidden_neurons_1, init_num_hidden_neurons_2)
            # num_hidden_neurons_1=50
            # num_hidden_neurons_2=50
            folds = [1,2,3,4,5]
            _y_score = []
            _y_true = []
            cm = []
            val_acc = []
            # for fold in folds:
            for fold in folds:
                # _x_train = np.load(data_dir  + class_type + '/x_train.npy')
                # _y_train = np.load(data_dir+ class_type + '/y_train.npy')
                # _x_test = np.load(data_dir+ class_type + '/x_test.npy')
                # _y_test = np.load(data_dir+ class_type + '/y_test.npy')
                train_list = np.load('../data/data_list/'+str(fold)+'_'+classification_type +'_list_train.npy')
                test_list = np.load('../data/data_list/'+str(fold)+'_'+classification_type +'_list_test.npy') 

                _x_train, _y_train,  _x_test, _y_test = data_generator(data_dir, classification_type, train_list, test_list)
                if ix == 0:
                    x_train, y_train,  x_test, y_test = _x_train, _y_train, _x_test, _y_test
                    percentage_cutoff = thresholds[ix]


                else:
                    threshold_1 = thresholds[ix-1]
                    percentage_cutoff = thresholds[ix]
                    prev_threshold = thresholds[ix-1]
                    # x_train, y_train, x_test, y_test = _x_train, _y_train, _x_test, _y_test
                    mask= np.load(mask_saving+ 'mask_'+str(thresholds[ix-1])+ '.npy')
                    # prev_best_model = TARGET_DIRECTORY + class_subset + '_best_model_' + str(prev_threshold) + '.h5'
                    # sensitivity_filename, mask = compute_deeplift_scores(data_dir, class_subset, prev_best_model, 0, 1, 0, mask, gpu_id, threshold_1, threshold_2)
                    # (X_,Y) = prepare_dataset_adni2(data_dir)
                    x_train = []
                    for sample in _x_train:
                        new_sample = np.multiply(sample, mask)[mask == 1]
                        x_train.append(new_sample)

                    x_train = np.array(x_train)
                    x_test = []
                    for sample in _x_test:
                        new_sample = np.multiply(sample, mask)[mask == 1]
                        x_test.append(new_sample)

                    y_train, y_test =  _y_train,  _y_test
                    # prev_best_model = TARGET_DIRECTORY + class_subset + '_best_model_' + str(prev_threshold) + '.h5'
                    # sensitivity_filename, mask = compute_deeplift_scores(data_dir, class_subset, prev_best_model, 0, 1, 0, mask, gpu_id, threshold_1, threshold_2)
                    # (X_,Y) = prepare_dataset_adni2(data_dir)
                    x_test = np.array(x_test)
                    y_train, y_test =  _y_train,  _y_test

                
                init_num_hidden_neurons_1 = init_num_hidden_neuron_1-ix
                init_num_hidden_neurons_2 = init_num_hidden_neuron_2-ix
                init_num_hidden_neurons_3 = init_num_hidden_neuron_3-ix
                
                seed(2)
                set_random_seed(3)
                from keras import backend as K
                K.clear_session()
                model = Sequential()
                model.add(Hadamard())
                # model.add(Dense(init_num_hidden_neurons_1, activation='relu', input_shape=np.shape(x_train[1])))
                model.add(Dense(init_num_hidden_neurons_1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(init_num_hidden_neurons_2, activation='relu'))
                # model.add(Dense(init_num_hidden_neurons_3, activation='relu'))
                model.add(Fixed_Layer())
                # model.add(Dense(init_num_hidden_neurons_3, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(num_classes))
                model.add(Activation('softmax')) 


                # model.summary()
                opt = optimizers.Adam()
                model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
                cp_callback = [early_stop] 
                history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        callbacks=cp_callback,
                        epochs=epochs,
                        verbose=0,                            
                        validation_data=(x_test, y_test)
                        )
                predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
                test_score = model.evaluate(x_test, y_test, verbose=0)
                # print('Test loss:', test_score[0])                

                valacc = max(history.history['val_acc'])
                val_acc.append(valacc)
                # best_epoch = history.history['val_acc'].index(max(history.history['val_acc']))
                print('<.......................................................>',ix, i, str(classification_type), str(contr_score), 'Fold_', str(fold), str(np.shape(x_train)))
                print('val_acc', valacc)
                print('Test accuracy:', test_score[-1])

                _cm = metrics.confusion_matrix(y_test.argmax(1), predictions.argmax(1))                
                _y_true.extend(y_test)
                _y_score.extend(predictions)
                cm.append(_cm)   
                # print(_cm)             




                if ix == 0:                
                    np.save(mask_saving+ 'mask_'+str(thresholds[ix])+ '.npy', mask)
                else:
                    X_nontest, Y_nontest, X_test, Y_test = _x_train, _y_train, _x_test, _y_test
                    num_features = int(np.sum(mask))
                    X_nontest, mapping = get_masked_data(X_nontest, mask)
                    X_test, _ = get_masked_data(X_test, mask)

                    X_nontest_length = len(X_nontest)
                    X_test_length = len(X_test)

                    Y_nontest = np.argmax(Y_nontest, axis=1)
                    Y_test = np.argmax(Y_test, axis=1)

                 
                    xs = x_test
                    ys = y_test                

                    if contr_score=='deeplift':
                        attr= get_contr_score_deeplift(model, xs, ys)
                    elif contr_score=='grad_input':
                        attr= get_contr_score_grad_input(model, xs, ys)
                    elif contr_score=='saliency':
                        attr= get_contr_score_saliency(model, xs, ys)
                    elif contr_score=='intgrad':
                        attr= get_contr_score_intgrad(model, xs, ys)
                    elif contr_score=='elrp':
                        attr= get_contr_score_elrp(model, xs, ys)
                    elif contr_score=='occlusion':
                        attr= get_contr_score_occlusion(model, xs, ys)
                   

                    scores = np.array(attr)
                    sum_scores = np.zeros(x_test[0].shape)
                    for score in scores:
                        sum_scores += np.square(score)

                    padded_sum_scores = get_padded_data(sum_scores, mapping)
                    full_matrix = np.zeros((116, 116))
                    full_matrix[np.triu_indices(116, 0)] = padded_sum_scores
                    full_matrix_T = full_matrix.T
                    full_matrix = full_matrix + full_matrix_T - np.diag(np.diag(full_matrix_T))
                    nodal_sensitivity = np.sum(np.absolute(full_matrix), axis = 0)
                    # print(nodal_sensitivity)
                    num_nodes = full_matrix.shape[0]

                    nodal_sensitivity_sort = np.sort(nodal_sensitivity)
                    threshold_val = nodal_sensitivity_sort[int((1.0-percentage_cutoff)*num_nodes)]

                    selected_nodes = np.argwhere(nodal_sensitivity > threshold_val)
                    selected_features_matrix = np.zeros(full_matrix.shape)

                    # print ('Important nodes: ', selected_nodes)
                    for node_1 in selected_nodes:
                        selected_features_matrix[node_1, :] = 1
                        selected_features_matrix[:, node_1] = 1
                    np.fill_diagonal(selected_features_matrix, 0)
                    
                    # print (selected_features_matrix.shape)
                    mask = remove_symmetry(selected_features_matrix)
                    np.save(mask_saving+ 'mask_'+str(thresholds[ix])+ '.npy', mask)
                    


                    
                    # print('_confusion_matrix.ravel()_confusion_matrix.ravel()',tn, fp, fn, tp)
                    # print('_confusion_matrix.ravel()_confusion_matrix.ravel()',sensitivity, specificity, f1, acc)

            if ix == 0:
                print('pass')
            else:
                np.savetxt(important_features + mode + '_scores_deeplift_' + classification_type + '_reduced_' + str(threshold_1) + '_t_' + str(percentage_cutoff) + '.csv', np.transpose(np.array(scores)), delimiter= ',')
                # print('Writing reshaped scores')
                np.savetxt(important_features + mode + '_scores_reshaped_' + classification_type + '_reduced_' + str(threshold_1) + '_t_' + str(percentage_cutoff) + '.csv', full_matrix, delimiter=",")

                selected_features_file = important_features + classification_type + '_deeplift_features_nodes_r_' + str(threshold_1) + '_t_' + str(percentage_cutoff) + '.csv'
                np.savetxt(selected_features_file, selected_features_matrix)
            

            y_true=[]
            for label in _y_true:
                y_true.append(label[1])

            y_score=[]
            for score in _y_score:
            #     print (axis1[1])
                y_score.append(score[1])

            _confusion_matrix = np.zeros([2,2])
            for c in cm:
                _confusion_matrix+=c
            _confusion_matrix=np.divide(_confusion_matrix, fold)  

            tn, fp, fn, tp = _confusion_matrix.ravel()
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            f1 = (2 * tp) / (2 * tp + fp + fn)
            test_acc = (tp+tn)/(tp+tn+fp+fn)
            validation_acc=np.mean(val_acc)
            print('All Fold validation_acc', validation_acc)
            print('All Fold test', test_acc)

            np.save(training_logs + 'y_true_'+str(i)+'.npy', y_true)
            np.save(training_logs + 'y_score_'+str(i)+'.npy', y_score)
            np.save(training_logs + '_confusion_matrix_'+str(i)+'.npy', _confusion_matrix)

            with open(training_logs + 'training_logs_train_acc_' + classification_type + '.csv', 'a') as out_stream:
                # out_stream.write(str(i)  + ', ' + str(sensitivity) + ', ' + str(specificity) + ', ' + str(f1) + ', ' + str(valacc) + ', ' + str(test_acc) + ', ' + str(np.shape(x_train)) + '\n')
                out_stream.write( str(valacc)  + ', ')
            with open(training_logs + 'training_logs_all_indices_' + classification_type + '.csv', 'a') as out_stream:
                out_stream.write( str(sensitivity) + ', ' + str(specificity) + ', ' + str(f1) + ', ' + str(valacc) + ', ' + str(test_acc) + ', ' + str(np.shape(x_train)) + '\n')
                # out_stream.write( str(test_acc)  + ', ')
            with open(training_logs + 'training_logs_test_acc_' + classification_type + '.csv', 'a') as out_stream:
                # out_stream.write(str(i)  + ', ' + str(sensitivity) + ', ' + str(specificity) + ', ' + str(f1) + ', ' + str(valacc) + ', ' + str(test_acc) + ', ' + str(np.shape(x_train)) + '\n')
                out_stream.write( str(test_acc)  + ', ')


