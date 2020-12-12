import keras
import keras.backend as k
import tensorflow as tf
import art.attacks.evasion.simba as sm

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.utils import load_mnist
from art.estimators.classification import KerasClassifier


def cnn_mnist_k(input_shape):
    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
    )

    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    return classifier


# Get session
session = tf.Session()
k.set_session(session)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

# Construct and train a convolutional neural network on MNIST using Keras
source = cnn_mnist_k(x_train.shape[1:])
source.fit(x_train, y_train, nb_epochs=5, batch_size=128)


adv_crafter = sm.SimBA(source)
