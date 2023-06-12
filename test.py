import os
import tensorflow as tf

# import tensorflowjs as tfjs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.tacotron import Tacotron

model = Tacotron(K=16, conv_dim=[128, 128])

optimizer = Adam()

model.build(
    input_shape=(None, None, 80),
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
)

model.save("export/model", save_format="tf")
# tfjs.converters.
# tfjs.converters.save_keras_model(model, '../drive/MyDrive/tacotron2/tensorflow2/tacotron')
