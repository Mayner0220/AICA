import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from setup import AUTOTUNE, BATCH_SIZE, IMAGE_SIZE, EPOCHS

# 경증 치매, 중증도 치매, 비 치매, 매우 경미한 치매
CLASS_NAMES = ["MildDementia", "ModerateDementia", "NonDementia", "VeryMildDementia"]
NUM_CLASSES = len(CLASS_NAMES)

def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
train_ds.class_names = CLASS_NAMES
train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds.class_names = CLASS_NAMES
val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)