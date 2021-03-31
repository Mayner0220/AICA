import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE
# IMAGE SIZE: 176 * 208
IMAGE_SIZE = [176, 208]

# Starting batch size: 16
# Starting epoch: 100
# 이 값들은 차후적으로 튜닝을 진행하면서 변화할 예정
BATCH_SIZE = 16
EPOCHS = 100

# 경증 치매, 중증도 치매, 비 치매, 매우 경미한 치매ㅋ
CLASS_NAMES = ["MildDementia", "ModerateDementia", "NonDementia", "VeryMildDementia"]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
train_ds.class_names = CLASS_NAMES

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds.class_names = CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)