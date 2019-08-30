import tensorflow as tf


def nn(input_shape, n_classes):
    """
    Function creating the model's graph in Keras.

    :param input_shape: shape of the model's input data (using Keras conventions)
    :param n_classes: n_classes to predict for the last dense layer
    :param kernel_size: kernel size of the first conv layer
    :param stride : stride_size of the first conv layer
    :return: Keras model instance
    """

    model = tf.keras.models.Sequential()

    # Step 1: First GRU Layer (≈4 lines)
    X_forward = tf.keras.layers.LSTM(units=128, activation='relu', return_sequences=True, input_shape=input_shape)  # GRU (use 128 units and return the sequences)
    X_backward = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, go_backwards=True)
    model.add(tf.keras.layers.Bidirectional(X_forward, backward_layer=X_backward))
    model.add(tf.keras.layers.Dropout(0.2))  # dropout (use 0.8)
    model.add(tf.keras.layers.BatchNormalization())  # Batch normalization

    # Step 4: Time-distributed dense layer (≈1 line)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="softmax")))  # time distributed (sigmoid)

    return model
