
from multiprocessing import freeze_support
import os
import numpy as np
import tensorflow as tf
import keras
from sklearn.neighbors import NearestNeighbors
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions
from src.AutoencoderRetrievalModel import AutoencoderRetrievalModel
from src.PretrainedModel import PretrainedModel
from src.AbstractAE import AbstractAE
from sklearn.decomposition import PCA

from keras.callbacks import EarlyStopping, ModelCheckpoint

class ImageRetrievalClass:
    def __init__(self, modelName, trainModel, parallel):
        self.modelName = modelName
        self.trainModel = trainModel
        self.parallel = parallel

            # Make paths
        self.dataTrainDir = os.path.join(os.getcwd(), "data", "train")
        self.dataTestDir = os.path.join(os.getcwd(), "data", "test")
        self.outDir = os.path.join(os.getcwd(), "output", self.modelName)
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        # Read images
        extensions = [".jpg", ".jpeg"]
        print("Reading train images from '{}'...".format(self.dataTrainDir))
        self.imgs_train = read_imgs_dir(self.dataTrainDir, extensions, parallel=parallel)
        print("Reading test images from '{}'...".format(self.dataTestDir))
        self.imgs_test = read_imgs_dir(self.dataTestDir, extensions, parallel=parallel)
        self.shape_img = self.imgs_train[0].shape
        print("Image shape = {}".format(self.shape_img))

        self.strategy = tf.distribute.MirroredStrategy()

    def buildModel(self):
        if self.modelName in ["simpleAE", "Resnet50AE", "stackedAE"]:
            info = {
                "shape_img": self.shape_img,
                "autoencoderFile": os.path.join(self.outDir, "{}_autoecoder.h5".format(self.modelName)),
                "encoderFile": os.path.join(self.outDir, "{}_encoder.h5".format(self.modelName)),
                "decoderFile": os.path.join(self.outDir, "{}_decoder.h5".format(self.modelName)),
                "checkpoint" : os.path.join(self.outDir,"{}_checkpoint.h5".format(self.modelName))
            }
            self.model = AutoencoderRetrievalModel(self.modelName, info)
            self.model.set_arch()

            self.shape_img_resize = self.model.getShape_img()
            self.input_shape_model = self.model.getInputshape()
            self.output_shape_model = self.model.getOutputshape()
        
        elif self.modelName in ["vgg19", "ResNet50v2", "IncepResNet"]:
            pretrainedModel = PretrainedModel(self.modelName,self.shape_img)
            self.model = pretrainedModel.buildModel()
            self.shape_img_resize, self.input_shape_model, self.output_shape_model = pretrainedModel.makeInOut()

        else:
            raise Exception("Invalid modelName!")

        # Print some model info
        print("input_shape_model = {}".format(self.input_shape_model))
        print("output_shape_model = {}".format(self.output_shape_model))

    def transform(self):
        
        class ImageTransformer(object):
            def __init__(self, shape_resize):
                self.shape_resize = shape_resize
            
            def __call__(self, img):
                img_transformed = resize_img(img, self.shape_resize)
                img_transformed = normalize_img(img_transformed)
                return img_transformed

        transformer = ImageTransformer(self.shape_img_resize)
        print("Applying image transformer to training images...")
        imgs_train_transformed = apply_transformer(self.imgs_train, transformer, parallel=self.parallel)
        print("Applying image transformer to test images...")
        imgs_test_transformed = apply_transformer(self.imgs_test, transformer, parallel=self.parallel)

        self.X_train = np.array(imgs_train_transformed).reshape((-1,) + self.input_shape_model)
        self.X_test = np.array(imgs_test_transformed).reshape((-1,) + self.input_shape_model)
        print(" -> X_train.shape = {}".format(self.X_train.shape))
        print(" -> X_test.shape = {}".format(self.X_test.shape))

    def train(self):
       

        if isinstance(self.model, AbstractAE):
            if self.trainModel:
                print('Number of devices: {}'.format(
                    self.strategy.num_replicas_in_sync))
                with self.strategy.scope():
                    self.model.compile(loss="binary_crossentropy", optimizer="adam")
                
                early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1,patience=6, min_delta=0.0001)
                checkpoint = ModelCheckpoint(
                        os.path.join(self.outDir,"{}_checkpoint.h5".format(self.modelName)),
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True)
                
                self.model.fit(self.X_train, n_epochs=30, batch_size=32,callbacks=[early_stopping, checkpoint])
                self.model.save_models()
            else:
                self.model.load_models(loss="binary_crossentropy", optimizer="adam")

    def predict(self):
        
        print("Inferencing embeddings using pre-trained model...")
        self.E_train = self.model.predict(self.X_train)
        E_train_flatten = self.E_train.reshape((-1, np.prod(self.output_shape_model)))
        self.E_test = self.model.predict(self.X_test)
        E_test_flatten = self.E_test.reshape((-1, np.prod(self.output_shape_model)))
        print(" -> E_train.shape = {}".format(self.E_train.shape))
        print(" -> E_test.shape = {}".format(self.E_test.shape))
        print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
        print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))

        return self.E_train, self.E_test

    def reconstuctionVisualize(self):
        if self.modelName in ["simpleAE", "Resnet50AE", "stackedAE"]:
            print("Visualizing database image reconstructions...")
            imgs_train_reconstruct = self.model.decoder.predict(self.E_train)
            if self.modelName == "simpleAE":
                imgs_train_reconstruct = imgs_train_reconstruct.reshape((-1,) + self.shape_img_resize)
            plot_reconstructions(self.imgs_train, 
                                 imgs_train_reconstruct,
                                 os.path.join(self.outDir, "{}_reconstruct.png".format(self.modelName)),
                                 range_imgs=[0, 255],
                                 range_imgs_reconstruct=[0, 1])

    def similarityCalculator(self,E_train):
        E_train_flatten = E_train.reshape((-1, np.prod(self.output_shape_model)))
        print("Fitting k-nearest-neighbour model on training images...")
        calculator = NearestNeighbors(n_neighbors=5, metric="cosine")
        calculator.fit(E_train_flatten)

        return calculator
    
    def retrieval(self,E_test, calculator):
        print("Performing image retrieval on test images...")
        
        E_test_flatten = E_test.reshape((-1, np.prod(self.output_shape_model)))
        for i, emb_flatten in enumerate(E_test_flatten):
            # find k nearest train neighbours
            _, indices = calculator.kneighbors([emb_flatten])
            img_query = self.imgs_test[i]  # query image
            imgs_retrieval = [self.imgs_train[idx]
                            for idx in indices.flatten()]  # retrieval images
                            
            outFile = os.path.join(
                self.outDir, "{}_retrieval_{}.png".format(self.modelName, i))
            plot_query_retrieval(img_query, imgs_retrieval, outFile)