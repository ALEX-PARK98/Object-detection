import numpy as np
import tensorflow as tf

def one_hot_encod(dataset):
    size = dataset.shape[-1]
    one_hot_shape = (size, 10)
    one_hot_y = np.zeros(one_hot_shape)
    one_hot_y[np.arange(size), dataset] = 1

    return one_hot_y.T


def flatten(dataset):
    size = dataset.shape[0]
    data_shape = (size, 784)

    return dataset.reshape(data_shape)


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        layer_input_size = layer.get("input_dim")
        layer_output_size = layer.get("output_dim")

        params_values["W" + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size
        ) * 0.1
        params_values["b" + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def relu_backward(dA, Z):
    ## make copy of dA dZ = np.array(dA, copy = True)
    dA[Z <= 0] = 0
    return dA


if __name__ == "__main__":
    ## mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ## process data
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    y_test = one_hot_encod(y_test)
    y_train = one_hot_encod(y_train)

    ## prepare layers
    nn_architecture = [
        {"input_dim": 784, "output_dim": 128, "activation": "relu"},
        {"input_dim": 128, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 10, "activation": "softmax"},
    ]

    params_values = init_layers(nn_architecture=nn_architecture)

    W1 = params_values.get("W1")
    b1 = params_values.get("b1")
    W2 = params_values.get("W2")
    b2 = params_values.get("b2")
    W3 = params_values.get("W3")
    b3 = params_values.get("b3")
    W4 = params_values.get("W4")
    b4 = params_values.get("b4")
    W5 = params_values.get("W5")
    b5 = params_values.get("b5")

    for i in range(10):
        ## forward propagation
        Z1 = np.dot(W1, x_train.T) + b1
        A1 = relu(
            np.array(Z1, copy=True)
        )

        Z2 = np.dot(W2, A1) + b2
        A2 = relu(
            np.array(Z2, copy=True)
        )

        Z3 = np.dot(W3, A2) + b3
        A3 = relu(
            np.array(Z3, copy=True)
        )

        Z4 = np.dot(W4, A3) + b4
        A4 = relu(
            np.array(Z4, copy=True)
        )

        Z5 = np.dot(W5, A4) + b5
        A5 = softmax(
            np.array(Z5, copy=True)
        )

        loss = - np.sum(y_train * np.log(A5)) / 60000
        print(f"loss: {loss}")

        accuracy = 0
        for i, j in zip(A5.T, y_test.T):
            if np.argmax(i) == np.argmax(j):
                accuracy += 1
        print(accuracy / 10000)

        ## backward propagation
        dZ5 = A5 - y_train
        dW5 = np.dot(dZ5, A4.T) / 60000
        db5 = np.sum(dZ5, axis=1, keepdims=True) / 60000

        dA4 = np.dot(W5.T, dZ5)
        dZ4 = relu_backward(
            Z4, np.array(dA4, copy=True)
        )
        dW4 = np.dot(dZ4, A3.T) / 60000
        db4 = np.sum(dZ4, axis=1, keepdims=True) / 60000

        dA3 = np.dot(W4.T, dZ4)
        dZ3 = relu_backward(
            Z3, np.array(dA3, copy=True)
        )
        dW3 = np.dot(dZ3, A2.T) / 60000
        db3 = np.sum(dZ3, axis=1, keepdims=True) / 60000

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = relu_backward(
            Z2, np.array(dA2, copy=True)
        )
        dW2 = np.dot(dZ2, A1.T) / 60000
        db2 = np.sum(dZ2, axis=1, keepdims=True) / 60000

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = relu_backward(
            Z1, np.array(dA1, copy=True)
        )
        dW1 = np.dot(dZ1, x_train) / 60000
        db1 = np.sum(dZ1, axis=1, keepdims=True) / 60000

        ## update weight and bias
        W1 = W1 - 0.2 * dW1
        b1 = b1 - 0.2 * db1
        W2 = W2 - 0.2 * dW2
        b2 = b2 - 0.2 * db2
        W3 = W3 - 0.3 * dW3
        b3 = b3 - 0.3 * db3
        W4 = W4 - 0.4 * dW4
        b4 = b4 - 0.4 * db4
        W5 = W5 - 0.5 * dW5
        b5 = b5 - 0.5 * db5

        loss = - np.sum(y_train * np.log(A5)) / 60000
        print(f"loss: {loss}")
        accuracy = 0
        for i, j in zip(A5.T, y_test.T):
            if np.argmax(i) == np.argmax(j):
                accuracy += 1
        print(accuracy / 10000)
