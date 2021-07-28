
from multiprocessing import freeze_support
import os
import numpy as np
import tensorflow as tf
# import keras
from sklearn.neighbors import NearestNeighbors
from src.CV_IO_utils import read_imgs_dir, read_imgs_list, read_one_image
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions
from src.AutoencoderRetrievalModel import AutoencoderRetrievalModel
from src.PretrainedModel import PretrainedModel
from src.AbstractAE import AbstractAE
from sklearn.decomposition import PCA

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ImageTransformer(object):
    def __init__(self, shape_resize):
        self.shape_resize = shape_resize
    
    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed


class ImageRetrievalClass:
    def __init__(self, modelName, trainModel, parallel):
        self.modelName = modelName
        self.trainModel = trainModel
        self.parallel = parallel
        self.strategy = tf.distribute.MirroredStrategy()

            # Make paths
        # self.dataTrainDir = os.path.join(os.getcwd(), "detected_data", "detected_from_train")
        # self.dataTestDir = os.path.join(os.getcwd(), "detected_data", "detected_from_test")
        self.outDir = os.path.join(os.getcwd(), "retrieval_output", self.modelName)
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

    def readTrainSet(self, detected_train):
        # Read images
        
        print("Reading train images")
        self.imgs_train = read_imgs_list(detected_train)
        self.shape_img = self.imgs_train[0].shape
      
        print("train image shape = {}".format(self.shape_img))

    def readTestSet(self, detected_test):
        print("Reading train images")
        self.imgs_test = read_one_image(detected_test)
        self.shape_img = self.imgs_test[0].shape
        print("test image shape = {}".format(self.shape_img))
    

    def buildModel(self):
        if self.modelName in ["simpleAE", "Resnet50AE", "stackedAE","vggAE"]:
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
        
        elif self.modelName in ["vgg19", "ResNet50v2", "IncepResNet", "EfficientNet"]:
            pretrainedModel = PretrainedModel(self.modelName,self.shape_img)
            self.model = pretrainedModel.buildModel()
            self.shape_img_resize, self.input_shape_model, self.output_shape_model = pretrainedModel.makeInOut()

        else:
            raise Exception("Invalid modelName!")

        # Print some model info
        print("input_shape_model = {}".format(self.input_shape_model))
        print("output_shape_model = {}".format(self.output_shape_model))

    def trainTransform(self):

        transformer = ImageTransformer(self.shape_img_resize)
        print("Applying image transformer to training images...")
        imgs_train_transformed = apply_transformer(self.imgs_train, transformer, parallel=self.parallel)

        self.X_train = np.array(imgs_train_transformed).reshape((-1,) + self.input_shape_model)
        print(" -> X_train.shape = {}".format(self.X_train.shape))
        return self.X_train

    def testTransform(self):

        transformer = ImageTransformer(self.shape_img_resize)
        print("Applying image transformer to test images...")
        imgs_test_transformed = apply_transformer(self.imgs_test, transformer, parallel=self.parallel)

        self.X_test = np.array(imgs_test_transformed).reshape((-1,) + self.input_shape_model)
        print(" -> X_test.shape = {}".format(self.X_test.shape))
        return self.X_test

    def train(self, X_train):
        

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
                
                self.model.fit(X_train, n_epochs=30, batch_size=32,callbacks=[early_stopping, checkpoint])
                self.model.save_models()
            else:
                self.model.load_models(loss="binary_crossentropy", optimizer="adam")

    def predictTrain(self, X_train):
        
        print("Inferencing embeddings using pre-trained model...")
        self.E_train = self.model.predict(X_train)
        # E_train_flatten = self.E_train.reshape((-1, np.prod(self.output_shape_model)))
        print(" -> E_train.shape = {}".format(self.E_train.shape))
        # print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
        return self.E_train

    def predictTest(self, X_test):
        
        print("Inferencing embeddings using pre-trained model...")
        self.E_test = self.model.predict(X_test)
        # E_test_flatten = self.E_test.reshape((-1, np.prod(self.output_shape_model)))
        print(" -> E_test.shape = {}".format(self.E_test.shape))
        # print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))
        return self.E_test
    
    def similarityCalculator(self, E_train_flatten):
        print("Fitting k-nearest-neighbour model on training images...")
        calculator = NearestNeighbors(n_neighbors=5, metric="cosine") # 팩토리와 AbstractCalculator 만들어서 OCP에 맞게 모듈화하기
        calculator.fit(E_train_flatten)

        return calculator


    # retrievalPool means detected-images queryed by tag 
    def retrieval(self,E_test_flatten, calculator, retrieval_imagePool):
        print("Performing image retrieval on test images...")

        # E_test_flatten = E_test.reshape((-1, np.prod(self.output_shape_model)))
        for i, emb_flatten in enumerate(E_test_flatten):
            # find k nearest train neighbours
            _, indices = calculator.kneighbors([emb_flatten])
            
            imgs_train = read_imgs_list(retrieval_imagePool)

            img_query = self.imgs_test[i]  # query image
            imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()]  # retrieval images
                            
            outFile = os.path.join(self.outDir, "{}_retrieval_{}.png".format(self.modelName, i))
            plot_query_retrieval(img_query, imgs_retrieval, outFile)