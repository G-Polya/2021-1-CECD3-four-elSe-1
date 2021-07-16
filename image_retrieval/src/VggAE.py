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
  

    def makeAutoencoder(self):
        self.input = Input(shape=self.shape_img)


        # encoder
        x = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same",name="Encoding_Conv2D_1")(self.input)
        x = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same",name="Encoding_Conv2D_2")(x) # 512 x 512 x 64
        x = MaxPool2D(pool_size=2,name="Encoding_MaxPool2D_1")(x)                                        # 256 x 256 x 64
        print("After Encoding_MaxPool2D_1... : ",x.shape)

        x = Conv2D(128, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_3")(x) # 256 256 128
        x = Conv2D(128, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_4")(x) 
        x = MaxPool2D(pool_size=2,name="Encoding_MaxPool2D_2")(x)                                        # 128 128 128
        print("After Encoding_MaxPool2D_2... : ",x.shape)


        x = Conv2D(256, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_5")(x) # 128 128 256
        x = Conv2D(256, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_6")(x)
        x = Conv2D(256, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_7")(x)
        x = MaxPool2D(pool_size=2,name="Encoding_MaxPool2D_3")(x)                                        # 64 64 256
        print("After Encoding_MaxPool2D_3... : ",x.shape)

        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_8")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_9")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_10")(x)
        x = MaxPool2D(pool_size=2,name="Encoding_MaxPool2D_4")(x)
        print("After Encoding_MaxPool2D_4... : ",x.shape)

        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_11")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_12")(x)
        x = Conv2D(512, kernel_size=(3,3), activation="relu",padding="same",name="Encoding_Conv2D_13")(x)
        self.encoded = MaxPool2D(pool_size=2,name="Encoding_MaxPool2D_5")(x) # decoder input, encoded output

        # decoder
        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_1")(self.encoded)
        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_2")(decoded_output)
        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_3")(decoded_output)
        decoded_output = UpSampling2D(size=2,name="Decoding_Upsampling_1")(decoded_output)

        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_4")(self.encoded)
        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_5")(decoded_output)
        decoded_output = Conv2D(512, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_6")(decoded_output)
        decoded_output = UpSampling2D(size=2,name="Decoding_Upsampling_2")(decoded_output)

        decoded_output = Conv2D(256, kernel_size=(1,1), activation="relu", padding="same",name="Decoding_Conv2D_7")(self.encoded)
        decoded_output = Conv2D(256, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_8")(decoded_output)
        decoded_output = Conv2D(256, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_9")(decoded_output)
        decoded_output = UpSampling2D(size=2,name="Decoding_Upsampling_3")(decoded_output)
        
        decoded_output = Conv2D(128, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_10")(decoded_output)
        decoded_output = Conv2D(128, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_11")(decoded_output)
        decoded_output = UpSampling2D(size=2,name="Decoding_Upsampling_4")(decoded_output)

        decoded_output = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_12")(decoded_output)
        decoded_output = Conv2D(64, kernel_size=(3,3), activation="relu", padding="same",name="Decoding_Conv2D_13")(decoded_output)
        self.decoded = UpSampling2D(size=2,name="Decoding_Upsampling_5")(decoded_output)

        self.autoencoder = Model(self.input, self.decoded)
        return self.autoencoder


    def makeEncoder(self):
        self.encoder = Model(self.input, self.encoded)
        return self.encoder

    def makeDecoder(self):
        output_encoder_shape = self.encoder.layers[-1].output_shape[1:]
        decoded_input = Input(shape=output_encoder_shape)
        decoded_output = self.autoencoder.layers[-19](decoded_input)
        for layer in self.autoencoder.layers[-18:]:
            decoded_output = layer(decoded_output)
            

        self.decoder = Model(decoded_input, decoded_output)
        return self.decoder


    def getInputshape(self):
        # return tuple([int(x) for x in self.encoder.input.shape[1:]])
        return tuple([int(x) for x in self.input.shape[1:]])

    def getOutputshape(self):
        # return tuple([int(x) for x in self.encoder.output.shape])
        return tuple([int(x) for x in self.encoded.shape[1:]])
