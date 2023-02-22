import tensorflow as tf
from tensorflow.keras.models import Model

def load_model(type):
    if type == "res":
        return tf.keras.models.load_model("resnet_mk.h5", compile=False)
    elif type == "xception":
        return tf.keras.models.load_model("xception_mk.h5", compile=False)
    else:
        return tf.keras.models.load_model("alinged_xception_mk.h5", compile=False)
