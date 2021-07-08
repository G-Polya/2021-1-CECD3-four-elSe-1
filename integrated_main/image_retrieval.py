"""

 image_retrieval.py  (author: Anson Wong / git: ankonzoid)

 We perform image retrieval using transfer learning on a pre-trained
 VGG image classifier. We plot the k=5 most similar images to our
 query images, as well as the t-SNE visualizations.

"""
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


from keras.callbacks import EarlyStopping, ModelCheckpoint


def image_retrieval(modelName="ResNet50v2",trainModel=True, parallel=False):
    # Run mode: (autoencoder -> simpleAE, convAE) or (transfer learning -> vgg19)
    modelName = "stackedAE"  # try: "simpleAE", "convAE", "vgg19" , "IncepResNet"
    trainModel = True
    parallel = False  # use multicore processing

    # Make paths
    dataTrainDir = os.path.join(os.getcwd(), "detected_data", "detected_from_train")
    dataQueryDir = os.path.join(os.getcwd(), "detected_data", "detected_from_test")
    outDir = os.path.join(os.getcwd(), "retrieval_output", modelName)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Read images
    extensions = [".jpg", ".jpeg"]
    print("Reading train images from '{}'...".format(dataTrainDir))
    imgs_train = read_imgs_dir(dataTrainDir, extensions, parallel=parallel)
    print("Reading test images from '{}'...".format(dataQueryDir))
    imgs_test = read_imgs_dir(dataQueryDir, extensions, parallel=parallel)
    shape_img = imgs_train[0].shape
    print("Image shape = {}".format(shape_img))

    strategy = tf.distribute.MirroredStrategy()

    # Build models
    if modelName in ["simpleAE", "convAE", "stackedAE"]:
     # Set up autoencoder
        info = {
            "shape_img": shape_img,
            "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
            "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
            "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
            "checkpoint" : os.path.join(outDir,"{}_checkpoint.h5".format(modelName))
        }
        model = AutoencoderRetrievalModel(modelName, info)
        model.set_arch()

        shape_img_resize = model.getShape_img()
        input_shape_model = model.getInputshape()
        output_shape_model = model.getOutputshape()
        
        n_epochs = 30

    elif modelName in ["vgg19", "ResNet50v2", "IncepResNet"]:
        pretrainedModel = PretrainedModel(modelName,shape_img)
        model = pretrainedModel.buildModel()
        shape_img_resize, input_shape_model, output_shape_model = pretrainedModel.makeInOut()
       


    else:
        raise Exception("Invalid modelName!")

    # Print some model info
    print("input_shape_model = {}".format(input_shape_model))
    print("output_shape_model = {}".format(output_shape_model))

    # Apply transformations to all images
    class ImageTransformer(object):

        def __init__(self, shape_resize):
            self.shape_resize = shape_resize

        def __call__(self, img):
            img_transformed = resize_img(img, self.shape_resize)
            img_transformed = normalize_img(img_transformed)
            return img_transformed

    transformer = ImageTransformer(shape_img_resize)
    print("Applying image transformer to training images...")
    imgs_train_transformed = apply_transformer(
        imgs_train, transformer, parallel=parallel)
    print("Applying image transformer to test images...")
    imgs_test_transformed = apply_transformer(
        imgs_test, transformer, parallel=parallel)

    # Convert images to numpy array
    X_train = np.array(imgs_train_transformed).reshape(
        (-1,) + input_shape_model)
    X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
    print(" -> X_train.shape = {}".format(X_train.shape))
    print(" -> X_test.shape = {}".format(X_test.shape))

    # Train (if necessary)
    if isinstance(model, AbstractAE):
        if trainModel:
            print('Number of devices: {}'.format(
                strategy.num_replicas_in_sync))
            with strategy.scope():
                model.compile(loss="binary_crossentropy", optimizer="adam")
            
            early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1,patience=6, min_delta=0.0001)
            checkpoint = ModelCheckpoint(
                    os.path.join(outDir,"{}_checkpoint.h5".format(modelName)),
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True)
            
            model.fit(X_train, n_epochs=n_epochs, batch_size=32,callbacks=[early_stopping, checkpoint])
            model.save_models()
        else:
            model.load_models(loss="binary_crossentropy", optimizer="adam")
 

    # Create embeddings using model
    print("Inferencing embeddings using pre-trained model...")
    E_train = model.predict(X_train)
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
    E_test = model.predict(X_test)
    E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
    print(" -> E_train.shape = {}".format(E_train.shape))
    print(" -> E_test.shape = {}".format(E_test.shape))
    print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
    print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))

    # Make reconstruction visualizations
    if modelName in ["simpleAE", "convAE", "stackedAE"]:
        print("Visualizing database image reconstructions...")
        imgs_train_reconstruct = model.decoder.predict(E_train)
        if modelName == "simpleAE":
            imgs_train_reconstruct = imgs_train_reconstruct.reshape(
                (-1,) + shape_img_resize)
        plot_reconstructions(imgs_train, imgs_train_reconstruct,
                             os.path.join(
                                 outDir, "{}_reconstruct.png".format(modelName)),
                             range_imgs=[0, 255],
                             range_imgs_reconstruct=[0, 1])

    # Fit kNN model on training images
    print("Fitting k-nearest-neighbour model on training images...")
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(E_train_flatten)

    # Perform image retrieval on test images
    print("Performing image retrieval on test images...")
    for i, emb_flatten in enumerate(E_test_flatten):
        # find k nearest train neighbours
        _, indices = knn.kneighbors([emb_flatten])
        img_query = imgs_test[i]  # query image
        imgs_retrieval = [imgs_train[idx]
                          for idx in indices.flatten()]  # retrieval images
        outFile = os.path.join(
            outDir, "{}_retrieval_{}.png".format(modelName, i))
        plot_query_retrieval(img_query, imgs_retrieval, outFile)

    # Plot t-SNE visualization
    # print("Visualizing t-SNE on training images...")
    # outFile = os.path.join(outDir, "{}_tsne.png".format(modelName))
    # plot_tsne(E_train_flatten, imgs_train, outFile)


if __name__ == "__main__":
    freeze_support()
    image_retrieval()
