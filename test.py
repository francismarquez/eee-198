import os

import utils
from utils import predict_tflite

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics
import time

def test(args):
    task = args.task
    if task == 'diagnostic':
        num_classes = 44
    elif task == 'subdiagnostic': 
        num_classes = 23
    else:
        num_classes = 5

    X_val = np.load(args.x_val)
    y_val = np.load(args.y_val)
    model = tf.keras.models.load_model(args.model)
    modelSplit = (args.model.split('/')[2]).split('_')[0]
    # print(modelSplit)
    # exit()
    model_no_quant_tflite = args.model + modelSplit + '.tflite'
    model_tflite = args.model + modelSplit + '_quantized.tflite'

    # Original Model
    y_val_pred = model.predict(X_val)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_val.flatten(), y_val_pred.flatten())
    auc = sklearn.metrics.roc_auc_score(y_val,y_val_pred, average='macro')
    hamming_loss = sklearn.metrics.hamming_loss(y_val, y_val_pred.round())
    f1_score = sklearn.metrics.f1_score(y_val,  y_val_pred.round(), average='samples')   

    print('Tensorflow Original Model')
    print(task + ' task')
    print('\nAUC:', auc)
    print('Hamming Loss:', hamming_loss)
    print('F1 Score:', f1_score)

    # Latency Test
    start_bigmodel = time.time()
    y_test_pred_tf = model.predict(X_val[0].reshape((1,1000,-1)))
    end_bigmodel = time.time()
    print('\nOriginal Model Latency')
    print(end_bigmodel - start_bigmodel, 'seconds')

    start_tflitemodel = time.time()
    y_test_pred_nonquantized_tflite = predict_tflite(model_no_quant_tflite, X_val[0].reshape((1,1000,-1)))
    end_tflitemodel = time.time()
    print('\nTFLite Latency')
    print(end_tflitemodel - start_tflitemodel, 'seconds')

    start_quantizedmodel = time.time()
    y_test_pred_quantized_tflite = predict_tflite(model_tflite, X_val[0].reshape((1,1000,-1)))
    end_quantizedmodel = time.time()
    print('\nTFLite Quantized Latency')
    print(end_quantizedmodel - start_quantizedmodel, 'seconds')

    y_test_pred_tf = model.predict(X_val)
    y_test_pred_nonquantized_tflite = predict_tflite(model_no_quant_tflite, X_val)
    y_test_pred_quantized_tflite = predict_tflite(model_tflite, X_val)

    y_test_pred_nonquantized_tflite = np.array(y_test_pred_nonquantized_tflite).reshape(-1, num_classes)
    y_test_pred_quantized_tflite = np.array(y_test_pred_quantized_tflite).reshape(-1, num_classes)

    # Non-quantized Model
    fpr_nonquantized, tpr_nonquantized, threshold = sklearn.metrics.roc_curve(y_val.flatten(), y_test_pred_nonquantized_tflite.flatten())
    auc_tflite_nonquantized = sklearn.metrics.roc_auc_score(y_val,y_test_pred_nonquantized_tflite, average='macro')
    hamming_loss = sklearn.metrics.hamming_loss(y_val, y_test_pred_nonquantized_tflite.round())
    f1_score = sklearn.metrics.f1_score(y_val, y_test_pred_nonquantized_tflite.round(), average='samples')

    print('TF Lite Non-Quantized Model')
    print(task + ' task')
    print('\nAUC:', auc_tflite_nonquantized)
    print('Hamming Loss:', hamming_loss)
    print('F1 Score:', f1_score)

    # Quantized Model
    fpr_quantized, tpr_quantized, threshold = sklearn.metrics.roc_curve(y_val.flatten(), y_test_pred_quantized_tflite.flatten())
    auc_tflite_quantized = sklearn.metrics.roc_auc_score(y_val,y_test_pred_quantized_tflite, average='macro')
    hamming_loss = sklearn.metrics.hamming_loss(y_val, y_test_pred_quantized_tflite.round())
    f1_score = sklearn.metrics.f1_score(y_val, y_test_pred_quantized_tflite.round(), average='samples')

    print('TF Lite Quantized Model')
    print(task + ' task')
    print('\nAUC:', auc_tflite_quantized)
    print('Hamming Loss:', hamming_loss)
    print('F1 Score:', f1_score)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr, label='TF (area = {:.3f})'.format(auc))
    plt.plot(fpr_quantized, tpr_quantized, label='TFLite Quantized (area = {:.3f})'.format(auc_tflite_quantized))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(' ROC curve on Test Data')
    plt.legend(loc='best')
    

    # File Size
    size_tf = utils.get_size(args.model)
    size_nonquan_tflite = os.stat(model_no_quant_tflite).st_size
    size_tflite = os.stat(model_tflite).st_size

    print("TF:\n % 2.3f" %(size_tf/1048576.) + 'MB\n')
    print("TFLite Non-quantized:\n % 2.3f" %(size_nonquan_tflite/1048576.) + 'MB\n')
    print("TFLite Quantized:\n % 2.3f" %(size_tflite/1048576.) + 'MB\n')

    plt.show()

if __name__ == '__main__':
    args = utils.get_args(
        description='Given trained model, load it and test validation dataset according to described metrics. Use -h for more deatail.',
        is_test=True, 
        is_train=False,
        is_demo=False)
    test(args)
