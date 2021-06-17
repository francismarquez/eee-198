# https://arxiv.org/pdf/1611.06455v4.pdf

import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class CNN1D:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)
        
    def build_model(self, input_shape, num_classes):
        model = keras.Sequential()
    
        # flatten/reshape because when multivariate all should be on the same axis 
        model.add(layers.Input(input_shape))
        model.add(keras.layers.Conv1D(filters=8,kernel_size=7,padding='valid',activation='sigmoid'))
        model.add(keras.layers.AveragePooling1D(pool_size=3))

        model.add(layers.Conv1D(filters=16,kernel_size=7,padding='valid',activation='sigmoid'))
        model.add(layers.AveragePooling1D(pool_size=3))

        model.add(layers.Conv1D(filters=32,kernel_size=7,padding='valid',activation='sigmoid'))
        model.add(layers.AveragePooling1D(pool_size=3))

        model.add(layers.Conv1D(filters=64,kernel_size=7,padding='valid',activation='sigmoid'))
        model.add(layers.AveragePooling1D(pool_size=3))

        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes, activation='sigmoid'))

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='AUC', factor=0.1, patience=1, verbose=1, mode='max',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),\
            optimizer=\
            tf.keras.optimizers.Adam(learning_rate=0.001),\
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5),\
                # tf.keras.metrics.RecallAtPrecision(name='Recall', precision=0.5),
                # tf.keras.metrics.Precision(name='Precision'),
                tf.keras.metrics.AUC(
                    num_thresholds=200,
                    curve="ROC",
                    summation_method="interpolation",
                    name="AUC",
                    dtype=None,
                    thresholds=None,
                    multi_label=True,
                    label_weights=None,
                )
            ]
        )

        self.callbacks = [reduce_lr]

        model.summary()

        return model
    
    def fit(self, x, y, batch_size, epochs, output_directory):

        hist = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks)
        
        self.model.save(output_directory)

        return hist