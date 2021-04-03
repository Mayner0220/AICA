import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from setup import AUTOTUNE, BATCH_SIZE, IMAGE_SIZE, EPOCHS
from data_load import NUM_CLASSES

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.input =  tf.keras.Input(shape=(*IMAGE_SIZE, 3))

        self.separalbe_conv16 = tf.keras.layers.SeparableConv2D(16, 3, activation="relu", padding="same")
        self.separalbe_conv32 = tf.keras.layers.SeparableConv2D(32, 3, activation="relu", padding="same")
        self.separalbe_conv64 = tf.keras.layers.SeparableConv2D(64, 3, activation="relu", padding="same")
        self.separalbe_conv128 = tf.keras.layers.SeparableConv2D(128, 3, activation="relu", padding="same")
        self.separalbe_conv256 = tf.keras.layers.SeparableConv2D(256, 3, activation="relu", padding="same")

        self.dese_cn = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
        self.dense_64 = tf.keras.layers.Dense(64, activation="relu")
        self.dense_128 = tf.keras.layers.Dense(128, activation="relu")
        self.dense_512 = tf.keras.layers.Dense(512, activation="relu")

        self.normalization = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x = self.input(x)

        x = self.separalbe_conv16(x)
        x = self.separalbe_conv16(x)
        x = self.maxpool(x)

        x = self.separalbe_conv32(x)
        x = self.separalbe_conv32(x)
        x = self.normalization(x)
        x = self.maxpool(x)

        x = self.separalbe_conv64(x)
        x = self.separalbe_conv64(x)
        x = self.normalization(x)
        x = self.maxpool(x)

        x = self.separalbe_conv128(x)
        x = self.separalbe_conv128(x)
        x = self.normalization(x)
        x = self.maxpool(x)

        x = tf.keras.layers.Dropout(x, 0.2)

        x = self.separalbe_conv256(x)
        x = self.separalbe_conv256(x)
        x = self.normalization(x)
        x = self.maxpool(x)

        x = tf.keras.layers.Dropout(x, 0.2)
        x = self.flatten(x)

        x = self.dense_512(x)
        x = self.normalization(x)
        x = tf.keras.layers.Dropout(x, 0.7)

        x = self.dense_128(x)
        x = self.normalization(x)
        x = tf.keras.layers.Dropout(x, 0.5)

        x = self.dense_64(x)
        x = self.normalization(x)
        x = tf.keras.layers.Dropout(x, 0.3)

        x = self.dese_cn(x)

        return x

model = Model()