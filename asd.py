import tensorflow as tf
import tensorflow.keras as keras

def model():
    txt_layer = keras.layers.Embedding(input_dim=1, output_dim=2, input_length=45,
                                       trainable=False, name='Embedding')