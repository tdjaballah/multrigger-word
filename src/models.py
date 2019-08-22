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

    X_input = tf.keras.layers.Input(shape=input_shape)

    # Step 1: First GRU Layer (≈4 lines)
    X = tf.keras.layers.GRU(units=256, return_sequences=True)(X_input)  # GRU (use 128 units and return the sequences)
    X = tf.keras.layers.Dropout(0.2)(X)  # dropout (use 0.8)
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization

    # Step 4: Time-distributed dense layer (≈1 line)
    X = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="softmax"))(X)  # time distributed (sigmoid)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)

    print(model.summary())

    return model
