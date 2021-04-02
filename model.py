import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from setup import AUTOTUNE, BATCH_SIZE, IMAGE_SIZE, EPOCHS

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv32 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")
        # self.conv64 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        # self.conv128 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")

        self.sep_conv32 = tf.keras.layers.SeparableConv2D(32, 3, activation="relu", padding="same")
        self.sep_conv64 = tf.keras.layers.SeparableConv2D(64, 3, activation="relu", padding="same")
        self.sep_conv128 = tf.keras.layers.SeparableConv2D(128, 3, activation="relu", padding="same")

    def call(self, input):
        pass