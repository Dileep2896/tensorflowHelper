import zipfile
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


class TensorFlowHelper:

    @staticmethod
    def extract_zipFiles(file_name: str):
        """
        Extract any zip file/folder
        :param file_name: Provide the file name in String (with full path)
        """
        zip_ref = zipfile.ZipFile(file_name)
        zip_ref.extractall()
        zip_ref.close()

    @staticmethod
    def walk_through_directories(file_name: str):
        """
        Walks through and print all the files folders in a directory.
        :param file_name: Provide the file name in String (with full path)
        """
        for dirPath, dirNames, filesNames in os.walk(file_name):
            print(f"There are {len(dirNames)} directories and {len(filesNames)} files in '{dirPath}'")

    @staticmethod
    def plot_loss_curve(history):
        """
        Returns separate loss curves for training and validation metrics.
        :param history: Tensorflow History object.
        :return: Plots of training/validation loss and accuracy metrics.
        """
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]

        epochs = range(len(history.history["loss"]))

        # Plot loss
        plt.plot(epochs, loss, label="training_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label="training_accuracy")
        plt.plot(epochs, val_accuracy, label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

    @staticmethod
    def tl_create_model(model_url, num_classes=10, Image_shape=(224, 224)):
        """
        Takes a TensorFlow Hub URL and create a Keras Sequential model with it.
        Used for transfer learning...
        For Image classification

        :param model_url: A TensorFlow Hub feature extraction URL.
        :param num_classes: Number of output neurons in the output layer,
                            should be equal to number of target classes, default 10.
        :param Image_shape: Image shape for the input layer
        :return: An un compiled Keras Sequential model with model_url as feature extractor
                 layer and Dense output layer with num_classes output neurons.
        """

        # Download the pretrained model and save it KerasLayer
        feature_extractor_layer = hub.KerasLayer(model_url,
                                                 trainable=False,  # freeze the already learned patterns
                                                 name="feature_extractor_model",
                                                 input_shape=Image_shape + (3,))

        # Create our own model
        model = tf.keras.Sequential([
            feature_extractor_layer,
            layers.Dense(num_classes, activation="softmax", name="output_layer")
        ])

        return model
