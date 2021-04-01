import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf

strategy = tf.distribute.get_strategy()

print("Number of replicas: {0}".format(strategy.num_replicas_in_sync))
print("TF version: {0}".format(tf.__version__))

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Starting batch size: 16
# Starting epoch: 100
# IMAGE SIZE: 176 * 208
# 이 값들은 차후적으로 튜닝을 진행하면서 변화할 예정
BATCH_SIZE = 16 * strategy
IMAGE_SIZE = [176, 208]
EPOCHS = 100