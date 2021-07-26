import tensorflow as tf
import numpy as np
from .AbstractAE import AbstractAE

class SimpleAE(AbstractAE):
    def __init__(self, info):
        super(SimpleAE, self).__init__(info)

    def makeAutoencoder(self):
        
        shape_img_flattened = (np.prod(list(self.shape_img)),)

        encode_dim = 128

        self.input = tf.keras.Input(shape=shape_img_flattened)
        self.encoded = tf.keras.layers.Dense(encode_dim, activation="relu")(self.input)
        self.decoded = tf.keras.layers.Dense(shape_img_flattened[0], activation='sigmoid')(self.encoded)

        self.autoencoder = tf.keras.Model(self.input, self.decoded)
        return self.autoencoder

    def makeEncoder(self):
        self.encoder = tf.keras.Model(self.input, self.encoded)
        return self.encoder

    def makeDecoder(self):
        self.autoencoder = self.makeAutoEncoder()
        # input_autoencoder_shape = self.autoencoder.layers[0].input_shape[1:]
        # output_autoencoder_shape = self.autoencoder.layers[-1].output_shape[1:]

        # input_encoder_shape = self.encoder.layers[0].input_shape[1:]
        output_encoder_shape = self.encoder.layers[-1].output_shape[1:]

        decoded_input = tf.keras.Input(shape=output_encoder_shape)
        decoded_output = self.autoencoder.layers[-1](decoded_input)

        self.decoder = tf.keras.Model(decoded_input, decoded_output)
        return self.decoder

    def getInputshape(self):
        return (self.input.shape[1],)

    def getOutputshape(self):
        return (self.encoded.shape[1],)


