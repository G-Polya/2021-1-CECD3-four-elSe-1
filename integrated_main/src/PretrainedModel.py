import numpy as np
import tensorflow as tf
# import keras

class PretrainedModel:
    def __init__(self, modelName,shape_img):
        self.modelName = modelName
        self.shape_img = shape_img
        self.model = None

    def buildModel(self):
        print(f"Loading {self.modelName} pre-trained model...")
        if self.modelName == "vgg19":
            # print("Loading VGG19 pre-trained model...")
            self.model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False,input_shape=self.shape_img)
        elif self.modelName == "IncepResNet":
            # print("Loading IncepResNet pre-trained model...")
            self.model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=self.shape_img)
        elif self.modelName == "ResNet50v2":
            # print("Loading ResNet50v2 pre-trained model...")
            self.model = tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet", include_top=False, input_shape=self.shape_img)
        elif self.modelName=="EfficientNet":
            # print("Loading EfficientNet pre-trained model...")
            self.model = tf.keras.applications.efficientnet.EfficientNetB4(weights="imagenet", include_top=False, input_shape=self.shape_img)
            # self.model = tf.keras.applications.efficientnet.EfficientNetB7(weights="imagenet", include_top=False, input_shape=self.shape_img)
        elif self.modelName=="MobileNetV3":
            
            self.model = tf.keras.applications.MobileNetV3Small(weights="imagenet", include_top=False, input_shape=self.shape_img)
        elif self.modelName=="MobileNetV2":
            self.model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, input_shape=self.shape_img)

        # self.model.summary()
        
        return self.model


    def makeInOut(self):
        shape_img_resize = tuple([int(x) for x in self.model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in self.model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in self.model.output.shape[1:]])
        n_epochs = None
        return shape_img_resize, input_shape_model,output_shape_model
