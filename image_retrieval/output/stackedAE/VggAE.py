# https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
from keras.layers.convolutional import Conv2DTranspose, UpSampling2D
from .AbstractAE import AbstractAE
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.layers import Subtract

class VggAE(AbstractAE):
    def __init__(self,info):
        super(VggAE, self).__init__(info)
  

    def vim(self):
        self.input = Input(shape=self.shape_img)


        # encoder
        x = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same")(self.input)
        x = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=2)(x)

        x = Conv2D(128, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(128, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = MaxPooling2D(pool_size=2)(x)
        
        x = Conv2D(256, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(256, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(256, kernel_size=(1,1), activation="relu",padding="same")(x)
        x = MaxPooling2D(pool_size=2)(x)

        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(512, kernel_size=(1,1), activation="relu",padding="same")(x)
        x = MaxPooling2D(pool_size=2)(x)

        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same")(x)
        x = Conv2D(512, kernel_size=(1,1), activation="relu",padding="same")(x)
        self.encoded = MaxPooling2D(pool_size=2)(x) # decoder input, encoded output

        # decoder
        decoded_output = Conv2DTranspose(512, kernel_size=(1,1), activation="relu", padding="same")(self.encoded)
        decoded_output = Conv2DTranspose(512, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = Conv2DTranspose(512, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = UpSampling2D(size=2)(decoded_output)

        decoded_output = Conv2DTranspose(512, kernel_size=(1,1), activation="relu", padding="same")(self.encoded)
        decoded_output = Conv2DTranspose(512, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = Conv2DTranspose(512, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = UpSampling2D(size=2)(decoded_output)

        decoded_output = Conv2DTranspose(256, kernel_size=(1,1), activation="relu", padding="same")(self.encoded)
        decoded_output = Conv2DTranspose(256, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = Conv2DTranspose(256, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = UpSampling2D(size=2)(decoded_output)
        
        decoded_output = Conv2DTranspose(128, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = Conv2DTranspose(128, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = UpSampling2D(size=2)(decoded_output)

        decoded_output = Conv2DTranspose(64, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        decoded_output = Conv2DTranspose(64, kernel_size=(3,3), activation="relu", padding="same")(decoded_output)
        self.decoded = UpSampling2D(size=2)(decoded_output)

        self.autoencoder = Model(self.input, self.decoded)
        return self.autoencoder


    def makeEncoder(self):
        self.encoder = Model(self.input, self.encoded)
        return self.encoder

    def makeDecoder(self):
        self.decoder = Model(self.encoded, self.decoded)
        return self.decoder


    def getInputshape(self):
        # return tuple([int(x) for x in self.encoder.input.shape[1:]])
        return tuple([int(x) for x in self.input.shape[1:]])

    def getOutputshape(self):
        # return tuple([int(x) for x in self.encoder.output.shape])
        return tuple([int(x) for x in self.encoded.shape[1:]])
