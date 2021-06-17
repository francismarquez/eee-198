import os

import utils
from nn_arch import resnet1d, mlp1d, cnn1d, fcn1d

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# import sklearn.metrics

def setup(args):
    input_shape = (1000,12)
    task = args.task
    if task == 'diagnostic':
        num_classes = 44
    elif task == 'subdiagnostic': 
        num_classes = 23
    else:
        num_classes = 5
    
    NN_ARCH = args.network_arch
    if NN_ARCH == 'MLP-1D':
        model = mlp1d.MLP1D(input_shape, num_classes)
    elif NN_ARCH == 'TimeCNN': 
        model = cnn1d.CNN1D(input_shape, num_classes)
    elif NN_ARCH == 'ResNet-1D': 
        model = resnet1d.ResNet1D(input_shape, num_classes)
    else:
        model = fcn1d.FCN1D(input_shape, num_classes)

    MODELS_DIR = './models/' + NN_ARCH + '_' + str(task)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    total_params = model.model.count_params()
    trainable_count = np.sum([keras.backend.count_params(w) for w in model.model.trainable_weights])

    with open(MODELS_DIR + '/' + NN_ARCH + ' records.txt', 'w') as f:
        f.write('---------\n')    
        f.write('Total params:\n')
        f.write(str(total_params) + '\n\n')
        f.write('Trainable params:\n')
        f.write(str(trainable_count) + '\n\n')

    return model, task, MODELS_DIR, NN_ARCH

def prep_data(task):
    # datafolder='./data/ptbxl/'
    datafolder = args.data_folder
    # outputfolder='./output/'
    outputfolder = args.output_folder

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder)

    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)

    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

    # 1-9 for training 
    X_train = data[labels.strat_fold < 10]
    y_train = Y[labels.strat_fold < 10]
    # 10 for validation
    X_val = data[labels.strat_fold == 10]
    y_val = Y[labels.strat_fold == 10]

    # save dataset to .npy files for test and demo access later on

    np.save(datafolder + 'raw_X_train', X_train)
    print('X_train saved!!')
    np.save(datafolder + 'y_train', y_train)
    print('y_train saved!!')
    np.save(datafolder + 'raw_X_val', X_val)
    print('X_val saved!!')
    np.save(datafolder + 'y_val', y_val)
    print('y_val saved!!')

    X_train, X_val = utils.preprocess_signals(X_train, X_val)

    np.save(datafolder + 'X_train', X_train)
    print('X_train saved!!')
    np.save(datafolder + 'X_val', X_val)
    print('X_val saved!!')
    # print("X_train.shape", X_train.shape)
    # print("y_train.shape", y_train.shape)
    # print("X_val.shape", X_val.shape)
    # print("y_val.shape", y_val.shape)
    
    return X_train, y_train, X_val, y_val

def train(args):
    if not args.init:
        X_train = np.load(args.data_folder + 'X_train.npy')
        y_train = np.load(args.data_folder + 'y_train.npy')
        model, task, MODELS_DIR, NN_ARCH = setup(args)

    else:
        model, task, MODELS_DIR, NN_ARCH = setup(args)
        X_train, y_train, _, _ = prep_data(task)

    # batch_size=64
    # epochs=30
    batch_size=args.batch_size
    epochs=args.epochs

    if tf.test.is_gpu_available:
        # GPU use for training   
        history = model.fit(X_train, y_train, batch_size, epochs, MODELS_DIR)
    else:
        print('Error in starting training using GPU. Using CPU instead.')
        history = model.fit(X_train, y_train, batch_size, epochs, MODELS_DIR)
        # exit()

    # TRAINING LOSS
    lossPlot = plt.figure(1)
    plt.plot(history.history['loss'], label='loss')
    plt.title(NN_ARCH + ' Training Loss Curve')
    plt.xlabel('Epoch')data/ptbxl/y_val.npy
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(MODELS_DIR + '/' + NN_ARCH + 'Training LOSS')

    # TRAINING AUC
    aucPlot = plt.figure(2)
    plt.plot(history.history['AUC'], label='AUC')
    plt.title(NN_ARCH + ' Training AUC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='lower right')
    plt.savefig(MODELS_DIR + '/' + NN_ARCH + 'Training AUC')

    plt.show()

if __name__ == '__main__':
    args = utils.get_args(
        description='Train neural network to classify 12-lead ECG data according to the SCP-ECG Standard. Use -h for more detail.',
        is_train=True,
        is_test=False, 
        is_demo=False)
    train(args)