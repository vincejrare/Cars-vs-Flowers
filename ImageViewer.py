from PIL import Image
import numpy as np
import glob


def read_images(folder):
    """
    Create a single list of grayscaled pictures of all jpg files in a specific folder

    Arguments:
    folder -- the address of the folder from which to take the pictures

    Returns:
    images -- the list of images
    """
    files = [file for file in glob.glob(folder + '*.jpg')]
    images = []
    for file_name in files:
        image = Image.open(file_name)
        image_gray = image.convert("L")
        images.append(np.asarray(image_gray))
    return images


def flatten_images(images):
    """
    Convert a list of images into an array of flattened pictures

    Arguments:
    images -- the list of images

    Returns:
    images_flatten -- the array of flattened images
    """
    images_array = np.asarray(images)
    images_flatten = images_array.reshape(images_array.shape[0], -1).T
    return images_flatten


def create_label(folder):
    """
    Create arrays of data and labels from a folder of pictures. Cars are labeled 1, and flowers are labeled 0.

    Arguments:
    folder -- the address of the folder containing pictures of cars and flowers

    Returns:
    X -- the array of data containing cars and flowers
    Y -- the associated array of labels
    """
    cars = read_images(folder+'car\\')
    cars_flatten = flatten_images(cars)
    cars_labels = np.ones((1, cars_flatten.shape[1]))
    flowers = read_images(folder+'flower\\')
    flowers_flatten = flatten_images(flowers)
    flowers_labels = np.zeros((1, flowers_flatten.shape[1]))
    x = np.append(cars_flatten, flowers_flatten, axis=1)
    y = np.append(cars_labels, flowers_labels, axis=1)
    return x, y


def initialize_data():
    """
    Create arrays containing the training data and the testing data, along with their respective labels

    Returns:
    x_train -- the array of training data
    y_train -- the array of labels for the training data
    x_test -- the array of testing data
    y_test -- the array of labels for the testing data
    """
    x_train, y_train = create_label('dataset\\cars_vs_flowers\\training_set\\')
    x_test, y_test = create_label('dataset\\cars_vs_flowers\\test_set\\')

    return x_train, y_train, x_test, y_test


x1, y1, x2, y2 = initialize_data()
print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)
