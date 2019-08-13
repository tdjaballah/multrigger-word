import tensorflow as tf


def trigger_model(input_shape, n_classes, kernel_size, stride):
    """
    Function creating the model's graph in Keras.

    :param input_shape: shape of the model's input data (using Keras conventions)
    :param n_classes: n_classes to predict for the last dense layer
    :param kernel_size: kernel size of the first conv layer
    :param stride : stride_size of the first conv layer
    :return: Keras model instance
    """

    X_input = tf.keras.layers.Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = tf.keras.layers.Conv1D(512, kernel_size=kernel_size, strides=stride)(X_input)  # CONV1D
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization
    X = tf.keras.layers.Activation('relu')(X)  # ReLu activation
    X = tf.keras.layers.Dropout(0.2)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = tf.keras.layers.GRU(units=512, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = tf.keras.layers.Dropout(0.2)(X)  # dropout (use 0.8)
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = tf.keras.layers.GRU(units=256, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = tf.keras.layers.Dropout(0.2)(X)  # dropout (use 0.8)
    X = tf.keras.layers.BatchNormalization()(X)  # Batch normalization

    # Step 4: Time-distributed dense layer (≈1 line)
    X = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="softmax"))(X)  # time distributed (sigmoid)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)

    print(model.summary())

    return model


def encode_model(kernel_size, stride):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(1024, kernel_size=kernel_size, strides=stride,
                                     bias_regularizer=tf.keras.regularizers.l2(.01)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())  # Batch normalization
    model.add(tf.keras.layers.Activation('relu'))  # ReLu activation

    model.add(tf.keras.layers.Conv1D(512, kernel_size=kernel_size, strides=stride,
                                     bias_regularizer=tf.keras.regularizers.l2(.01)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())  # Batch normalization
    model.add(tf.keras.layers.Activation('relu'))  # ReLu activation

    model.add(tf.keras.layers.Conv1D(256, kernel_size=kernel_size // 2, strides=stride // 2,
                                     bias_regularizer=tf.keras.regularizers.l2(.01)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())  # Batch normalization
    model.add(tf.keras.layers.Activation('relu'))  # ReLu activation

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='sigmoid'))

    return model


def siamese_model(input_shape, kernel_size, stride):

    X_input_1 = tf.keras.layers.Input(shape=input_shape)
    X_input_2 = tf.keras.layers.Input(shape=input_shape)

    model = encode_model(kernel_size, stride)

    encoded_1 = model(X_input_1)
    encoded_2 = model(X_input_2)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_1, encoded_2])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = tf.keras.models.Model(inputs=[X_input_1, X_input_2], outputs=prediction)

    print(siamese_net.summary())

    return siamese_net
