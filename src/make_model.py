import tensorflow as tf


def seq_model(input_shape, n_classes, kernel_size, stride):
    """
    Function creating the model's graph in Keras.

    :param input_shape: shape of the model's input data (using Keras conventions)
    :param n_classes: n_classes to predict for the last dense layer
    :param kernel_size: kernel size of the first conv layer
    :param stride : stride_size of the first conv layer
    :return: Keras model instance
    """

    tf.keras.backend.clear_session()

    X_input = tf.keras.layers.Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = tf.keras.layers.Conv1D(196, kernel_size=kernel_size, strides=stride)(X_input)  # CONV1D
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization
    X = tf.keras.layers.Activation('relu')(X)  # ReLu activation
    X = tf.keras.layers.Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = tf.keras.layers.GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = tf.keras.layers.Dropout(0.8)(X)  # dropout (use 0.8)
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = tf.keras.layers.GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = tf.keras.layers.Dropout(0.8)(X)  # dropout (use 0.8)
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization
    X = tf.keras.layers.Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)

    return model
