import os
import utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

diagnostic_classes = [
    "1 AV block",
    "2 AV block",
    "3 AV block",
    "Anterolateral Myocardial Infarction",
    "AMI",
    "Anterolateral Myocardial Infarction",
    "Anteroseptal Myocardial Infarction",
    "Complete Left Bundle Branch Block",
    "Complete Right Bundle Branch Block",
    "Digitalis-effect",
    "Electrolytic Disturbance or Drug (former EDIS)",
    "Incomplete Left Bundle Branch Block",
    "Inferolateral Myocardial Infarction",
    "Inferior Myocardial Infarction",
    "Subendocardial Injury in Anterolateral Leads",
    "Subendocardial Injury in Anteroseptal Leads",
    "Subendocardial Injury in Inferolateral Leads",
    "Subendocardial Injury in Inferior Leads",
    "Subendocardial Injury in Lateral Leads",
    "Inferoposterolateral Myocardial Infarction",
    "Inferoposterior Myocardial Infarction",
    "Incomplete Right Bundle Branch Block",
    "Ischemic in Anterolateral Leads",
    "Ischemic in Anterior Leads",
    "Ischemic in Anteroseptal Leads",
    "Ischemic in Inferolateral Leads",
    "Ischemic in Inferior Leads",
    "Ischemic in Lateral Leads",
    "Non-specific Ischemic",
    "Non-specific Intraventricular conduction disturbance (block)",
    "Left Anterior Fascicular Block",
    "Left Atrial Overload/Enlargement",
    "Lateral Myocardial Infarction",
    "Long QT-Interval",
    "Left Posterior Fascicular Block",
    "Left Ventricular Hypertrophy",
    "Non-diagnostic T Abnormalities",
    "NORMAL",
    "Non-specific ST Changes",
    "Posterior Myocardial Infarction",
    "Right Atrial Overload/Enlargement",
    "Right Ventricular Hypertrophy",
    "Septal hypertrophy",
    "Wolff-Parkinson-White syndrome"
]

subdiag_classes = ["AMI", "CLBBB", "CRBBB", "ILBBB", "IMI", "IRBBB", "ISCA", "ISCI", "ISC_", "IVCD", "LAFB/LPFB", "LAO/LAE", "LMI", "LVH", "NORM", "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP", "STTC", "WPW", "_AVB"]
superdiag_classes= ["CD", "HYP", "MI", "NORM", "STTC"]

def predict_tflite(tflite_model, x_test):
    # Prepare the test data
    x_test_ = x_test.copy()
    x_test_ = x_test_.astype(np.float32)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
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

    interpreter.set_tensor(input_details["index"], x_test_.reshape(1,1000,12))
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details["index"])
    ecgs.append(y_pred)

    # If required, dequantized the output layer (from integer to float)
    output_scale, output_zero_point = output_details["quantization"]

    if (output_scale, output_zero_point) != (0.0, 0):
        y_pred = y_pred.astype(np.float32)
        y_pred = (y_pred - output_zero_point) * output_scale

    return ecgs




def demo(args):

    # classes = None
    task = args.task
    if task == 'diagnostic':
        model = './models/FCN-1D_diagnostic/FCN-1D_quantized.tflite'
        classes = diagnostic_classes
    if task == 'subdiagnostic':
        # model = './models/FCN-1D_subdiagnostic/FCN-1D_quantized.tflite'
        classes = subdiag_classes
    elif task == 'superdiagnostic':
        # model = './models/FCN-1D_superdiagnostic/FCN-1D_quantized.tflite'
        classes = superdiag_classes

    input = np.genfromtxt(args.input, delimiter=',')
    input = (input - input.mean())/(input.std()) # standardize
    pred = np.array(predict_tflite(model, input))

    os.system("clear")
    print('\nDiagnosis is/are:')

    for idx, val in np.ndenumerate(pred.flatten()):
        if val > 0.5:
            print(classes[sum(idx)] + ' ' + str(round(val*100,3)) + '%')

if __name__ == '__main__':
    args = utils.get_args(
        description='Demo script of identifying present heart abnormality given a .csv file input. Use -h for more detail.',
        is_demo=True, 
        is_train=False,
        is_test=False)
    demo(args)