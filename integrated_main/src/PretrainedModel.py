import numpy as np
import tensorflow as tf
import keras

class PretrainedModel:
    def __init__(self, modelName,shape_img):
        self.modelName = modelName
        self.shape_img = shape_img
        self.model = None

    def buildModel(self):
        if self.modelName == "vgg19":
            print("Loading VGG19 pre-trained model...")
            self.model = keras.applications.VGG19(weights='imagenet', include_top=False,input_shape=self.shape_img)
        elif self.modelName == "IncepResNet":
            print("Loading IncepResNet pre-trained model...")
            self.model = keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=self.shape_img)
        elif self.modelName == "ResNet50v2":
            print("Loading ResNet50v2 pre-trained model...")
            self.model = keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=self.shape_img)
        
        self.model.summary()

        
        return self.model


    def makeInOut(self):
        shape_img_resize = tuple([int(x) for x in self.model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in self.model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in self.model.output.shape[1:]])
        n_epochs = None
        return shape_img_resize, input_shape_model,output_shape_model
