
#import keras
from .utils import split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from .RetrievalModelFactory import RetrievalModelFactory
from .AbstractRetrievalModel import AbstractRetrievalModel

"""
W
 autoencoder.py  (author: Anson Wong / git: ankonzoid)

"""
layers = tf.keras.layers

class AutoencoderRetrievalModel(AbstractRetrievalModel):

    def __init__(self, modelName, info):
        super(AutoencoderRetrievalModel, self).__init__(modelName,info)
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def getShape_img(self):
        return self.ae.getShape_img()

    def getInputshape(self):
        return self.ae.getInputshape()

    def getOutputshape(self):
        return self.ae.getOutputshape()

    # Train
    def fit(self, X, n_epochs=50, batch_size=256, callbacks=None):
        indices_fracs = split(fracs=[0.9, 0.1], N=len(X), seed=0)
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        self.autoencoder.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid),
                             callbacks=callbacks)

    # Inference
    def predict(self, X):
        return self.encoder.predict(X)

    # Set neural network architecture
    def set_arch(self):
        autoencoderFactory = RetrievalModelFactory()
        self.ae = autoencoderFactory.makeRetrievalModel(self.modelName,self.info)

        self.autoencoder = self.ae.makeAutoencoder()
        self.encoder = self.ae.makeEncoder()
        self.decoder = self.ae.makeDecoder()

        # Generate summaries
        print("\nautoencoder.summary():")
        print(self.autoencoder.summary())
        print("\nencoder.summary():")
        print(self.encoder.summary())
        print("\ndecoder.summary():")
        print(self.decoder.summary())


    # Compile
    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    # Load model architecture and weights
    def load_models(self, loss="binary_crossentropy", optimizer="adam"):
        print("Loading models...")
        self.autoencoder = tf.keras.models.load_model(
            self.info["autoencoderFile"])
        self.encoder = tf.keras.models.load_model(self.info["encoderFile"])
        self.decoder = tf.keras.models.load_model(self.info["decoderFile"])
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)
        self.decoder.compile(optimizer=optimizer, loss=loss)

    # Save model architecture and weights to file
    def save_models(self):
        print("Saving models...")
        self.autoencoder.save(self.info["autoencoderFile"])
        self.encoder.save(self.info["encoderFile"])
