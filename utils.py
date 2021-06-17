# https://github.com/helme/ecg_ptbxl_benchmarking

import os, sys, pickle, argparse
import pandas as pd
import numpy as np
import tqdm
import wfdb
import ast
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from nn_arch import resnet1d, mlp1d, cnn1d, fcn1d

# DATA PROCESSING STUFF

def setup():
    sampling_frequency = 100
    task = 'diagnostic'
    num_classes = 44
    NN_ARCH = 'resnet1d'
    input_shape = (1000,12)
    model = resnet1d.ResNet1D(input_shape, num_classes)

    # input1 = input("SAMPLING FREQUENCY:\n(1) 100\n(2) 200\n")
    input2 = input("TASK:\n (1) Diagnostic\n(2) Subdiagnostic\n(3) Superdiagnostic\n")
    input3 = input("NEURAL NET ARCHITECTURE:\n (1) MLP-1D\n(2) CNN-1D\n(3) ResNet-1D\n(4) FCN-1D\n")

    # if input1 == "1":
        #     sampling_frequency = 100
        # else:
        #     sampling_frequency = 500

    if input2 == "1":
        task = 'diagnostic'
        num_classes = 44
    elif input2 == "2": 
        task = 'subdiagnostic'
        num_classes = 23
    else:
        task = 'superdiagnostic'
        num_classes = 5

    if input3 == "1":
        NN_ARCH = 'MLP-1D'
        model = mlp1d.MLP1D(input_shape, num_classes)
    elif input3 == "2": 
        NN_ARCH = 'TimeCNN'
        model = cnn1d.CNN1D(input_shape, num_classes)
    elif input3 == "3": 
        NN_ARCH = 'ResNet-1D'
        model = resnet1d.ResNet1D(input_shape, num_classes)
    else:
        NN_ARCH = 'FCN-1D'
        model = fcn1d.FCN1D(input_shape, num_classes)


    return task, num_classes, NN_ARCH, model

def load_dataset(path, sampling_rate=100):
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda z: ast.literal_eval(z))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)
    return X, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save Label Binarizer
    # with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
    #     pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

def preprocess_signals(X_train, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    return apply_standardizer(X_train, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def predict_tflite(tflite_model, x_test):
  # Prepare the test data
  x_test_ = x_test.copy()
  x_test_ = x_test_.astype(np.float32)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path=tflite_model)
#   interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # print('input: ', input_details)
  # print('\n output: ', output_details)

  # If required, quantize the input layer (from float to integer)
  input_scale, input_zero_point = input_details["quantization"]
  if (input_scale, input_zero_point) != (0.0, 0):
    x_test_ = x_test_ / input_scale + input_zero_point
    x_test_ = x_test_.astype(input_details["dtype"])
  
  # Invoke the interpreter
  ecgs = []

  y_pred = np.empty(x_test_.size, dtype=output_details["dtype"])
  for i in range(len(x_test_)):
    final_new_x_test_ = np.reshape(x_test_[i], (1000, 12))

    interpreter.set_tensor(input_details["index"], [final_new_x_test_])
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details["index"])
    ecgs.append(y_pred)
  
  # If required, dequantized the output layer (from integer to float)
  output_scale, output_zero_point = output_details["quantization"]
  if (output_scale, output_zero_point) != (0.0, 0):
    y_pred = y_pred.astype(np.float32)
    y_pred = (y_pred - output_zero_point) * output_scale

  return ecgs

def get_size(path):
    #initialize the size
    total_size = 0
    
    #use the walk() method to navigate through directory tree
    for dirpath, dirnames, filenames in os.walk(path):
        for i in filenames:
            
            #use join to concatenate all the components of path
            f = os.path.join(dirpath, i)
            
            #use getsize to generate size in bytes and add it to the total size
            total_size += os.path.getsize(f)
    return total_size

def get_args(description, is_demo, is_train, is_test):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--task', choices={'diagnostic', 'subdiagnostic', 'superdiagnostic'}, required=is_train or is_demo, help='diagnostic (44 classes), sub-diagnostic (23 class), or super-diagnostic (5 classes)')

    # training
    parser.add_argument('--network-arch', choices={'MLP-1D', 'TimeCNN', 'ResNet-1D', 'FCN-1D'}, required=is_train, help='choose network architecture to use')
    parser.add_argument('--data-folder', type=str, required=is_train, help='path to dataset')
    parser.add_argument('--output-folder', type=str, required=is_train, help='path to desired output folder')
    parser.add_argument('--batch-size', type=int, required=is_train, help='training batch size')
    parser.add_argument('--epochs', type=int, required=is_train, help='training epochs')
    parser.add_argument('--init', action='store_true', help='save dataset array to .npy files')
    
    # test
    parser.add_argument('--model', type=str, required=is_test, help='X_val in .npy format')
    parser.add_argument('--x-val', type=str, required=is_test, help='X_val in .npy format')
    parser.add_argument('--y-val', type=str, required=is_test, help='y_val in .npy format')

    # demo
    parser.add_argument('--input', required=is_demo, help='input csv file')
    parser.add_argument('--show-chart', help='plot ECG waveform using matplotlib')

    args = parser.parse_args()
    return args