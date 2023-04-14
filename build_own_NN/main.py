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


def compute_loss(y,y_hat):
    m = y.shape[-1]

    return -(1 / m) * np.sum(y * np.log(y_hat))


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
    return np.maximum(0,Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)


def relu_backward(dA, Z):
    ## make copy of dA dZ = np.array(dA, copy = True)
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception("Non-supported activation function")

    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer, in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        active_function_curr = layer.get("activation")
        W_curr = params_values.get("W" + str(layer_idx))
        b_curr = params_values.get("b" + str(layer_idx))
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, active_function_curr)

        memory["A" + str(layer_idx)] = A_curr
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory


def softmax_backward_propagation(dZ_prev, A_prev):
    m = A_prev.shape[1]
    dW_curr = np.dot(dZ_prev, A_prev.T) / m
    db_curr = np.sum(dZ_prev, axis=1, keepdims=True) / m

    return dZ_prev, dW_curr, db_curr


def relu_backward_propagation(dZ_prev, A_prev, W_prev, Z_curr):
    m = A_prev.shape[1]
    dA_curr = np.dot(W_prev.T, dZ_prev)
    dZ_prev = relu_backward(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_prev, A_prev.T) / m
    db_curr = np.sum(dZ_prev, axis=1, keepdims=True) / m

    return dZ_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}

    dZ_prev = Y_hat - Y

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer.get("activation")

        A_prev = memory.get("A" + str(layer_idx_prev))
        Z_curr = memory.get("Z" + str(layer_idx_curr))

        if activ_function_curr == "softmax":
            dZ_prev, dW_curr, db_curr = softmax_backward_propagation(
                dZ_prev, A_prev
            )

        elif activ_function_curr == "relu":
            W_prev = params_values.get("W" + str(layer_idx_curr + 1))
            dZ_prev, dW_curr, db_curr = relu_backward_propagation(
                dZ_prev, A_prev, W_prev, Z_curr
            )

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate=0.1):
    ## enumerate(list, num) -> start with num
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values.get("dW" + str(layer_idx))
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values.get("db" + str(layer_idx))

    return params_values


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
    print(x_train.shape)
    print(y_train.shape)
    batch_X = list(
        np.reshape(
            x_train, (60, 1000, 784)
        )
    )
    batch_Y = np.split(y_train, 60, axis=1)

    print(batch_X[0].shape)
    print(batch_Y[0].shape)
#    for epoch in range(100):
#        A_curr, memory = full_forward_propagation(x_train.T, params_values, nn_architecture)
#        memory["A0"] = x_train.T
#        loss = compute_loss(y_train, memory.get("A5"))
#        print(f"loss: {loss}")
#        grads_values = full_backward_propagation(A_curr, y_train, memory, params_values, nn_architecture)
#        params_values = update(params_values, grads_values, nn_architecture, learning_rate=0.2)
#
#        A_curr, memory = full_forward_propagation(x_test.T, params_values, nn_architecture)
#        accuracy = 0
#        for i, j in zip(A_curr.T, y_test.T):
#            if np.argmax(i) == np.argmax(j):
#                accuracy += 1
#        print(f"accuracy: {accuracy/10000}%")

    for epoch in range(100):
        for x_train, y_train in zip(batch_X, batch_Y):
            A_curr, memory = full_forward_propagation(x_train.T, params_values, nn_architecture)
            memory["A0"] = x_train.T
            loss = compute_loss(y_train, memory.get("A5"))
            print(f"loss: {loss}")
            grads_values = full_backward_propagation(A_curr, y_train, memory, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate=0.2)

            A_curr, memory = full_forward_propagation(x_test.T, params_values, nn_architecture)
            accuracy = 0
            for i, j in zip(A_curr.T, y_test.T):
                if np.argmax(i) == np.argmax(j):
                    accuracy += 1
            print(f"accuracy: {accuracy/10000}%")
