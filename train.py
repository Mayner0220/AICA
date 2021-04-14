import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from model_v1 import AICA_v1

model = AICA_v1()
a = model.call()