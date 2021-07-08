<<<<<<< HEAD
# https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
from .AbstractAE import AbstractAE
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.layers import Subtract

class Resnet50AE(AbstractAE):
    def __init__(self,info):
        super(Resnet50AE, self).__init__(info)
  

    def makeAutoencoder(self):
        pass

       
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
=======
# https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
from .AbstractAE import AbstractAE
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.layers import Subtract

class Resnet50AE(AbstractAE):
    def __init__(self,info):
        super(Resnet50AE, self).__init__(info)
  

    def makeAutoencoder(self):
        pass

       
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
>>>>>>> f72ad1f9408fc1fe4d43d73d29c241a660890638
