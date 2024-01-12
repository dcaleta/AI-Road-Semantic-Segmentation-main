#from tensorflow.python.keras.layers import (
#     Input, Conv2D, BatchNormalization, Activation,
#    SeparableConv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
#)
#
#from keras.models import Model
#import tensorflow.python.keras as keras
import tensorflow as tf
from tensorflow.python import keras
from keras.layers  import Input
from keras.layers import Conv2D
from keras.layers  import BatchNormalization
from keras.layers  import Activation
from keras.layers  import SeparableConv2D
from keras.layers  import MaxPooling2D
from keras.layers  import Conv2DTranspose
from keras.layers  import UpSampling2D
from keras.models import Model

def get_model():
    inputs = Input(shape=(256, 256, 3), name="Input Image")
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x
    for filters in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])
        previous_block_activation = x
    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    outputs = Conv2D(13, 3, activation="softmax", padding="same")(x)

    return Model(inputs=[inputs], outputs=[outputs])

if __name__ == "__main__":
    model=get_model()
