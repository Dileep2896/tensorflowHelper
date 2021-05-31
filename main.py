import zipfile
import os
import matplotlib.pyplot as plt


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
        Walk through and print all the files folders in a directory.
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
