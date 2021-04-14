import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from setup import AUTOTUNE, BATCH_SIZE, IMAGE_SIZE

# 경증 치매, 중증도 치매, 비 치매, 매우 경미한 치매
CLASS_NAMES = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
NUM_CLASSES = len(CLASS_NAMES)

# 데이터셋의 label별로 나누어져 저장되어 있다.
# 즉, 이미 사전적으로 분류되어 있는 데이터셋이기에
# tf.keras의 전처리 기능을 이용해서 이미지를 loading
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
train_ds.class_names = CLASS_NAMES

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/train",
    labels='inferred',
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds.class_names = CLASS_NAMES