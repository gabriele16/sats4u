import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf

def swish(x):
    return keras.backend.sigmoid(x) * x

