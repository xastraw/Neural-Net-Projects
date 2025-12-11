import tensorflow as tf

def myNet():

    model = tf.keras.Sequential(layers = [
        # #images only have 1 color scale (greyscale)
        tf.keras.layers.InputLayer(shape=(1, 28, 28, 28)),

        #(3, 3, 3) slides a 3x3x3 cube over the height, depth, width
        # Using data_format channels first because the input has the channels (axis here) first instead of last
        tf.keras.layers.Conv3D(32, (3,3,3), activation= 'relu', data_format='channels_first'),
        tf.keras.layers.BatchNormalization(), #makes net less confident early on but more stable, it stabalizes the gradients
        tf.keras.layers.Conv3D(64, (2,2,2), activation= 'relu', data_format='channels_first'),
        tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), data_format='channels_first'),  #pool size of (2, 2, 2) changes the input shape of (28, 28, 28, 1) to (14, 14, 14, 1)
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Conv3D(128, (3,3,3), activation= 'relu', data_format='channels_first'),
        tf.keras.layers.BatchNormalization(), #drops the loss a fair bit
        tf.keras.layers.MaxPooling3D(pool_size=(2,2,2),data_format='channels_first'),
        
        
        #GlobalAveragePooling3D replaces flatten, hugely increases the accuracy (85%->92%) while also massively dropping the loss (0.9 -> 0.2)
        tf.keras.layers.GlobalAveragePooling3D(data_format='channels_first'), 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(11, activation='softmax', dtype='float32', name= 'output') #needs to be 11 because there are 10 classifications
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model