import tensorflow as tf
print("TensorFlow version:", tf.__version__)

if __name__ == "__main__":
    ## mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.0)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=1000)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy: ', test_acc)