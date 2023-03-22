import tensorflow as tf
import numpy as np

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 10, "activation": "sigmoid"},
]


## return W and b according to nn_architecture
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


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1-sig)


def relu_backward(dA, Z):
    ## make copy of dA dZ = np.array(dA, copy = True)
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
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

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(idx)] = Z_prev

    return A_curr, memory


def get_cost_value(Y_hat, Y):
    ## number of examples
    m = Y_hat.shape[1]

    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))

    ## numpy.squeeze() function is used when we want to remove single-dimensional entries from the shape of an array.
    return np.squeeze(cost)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0

    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)

    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    dW_curr = np.dot(dZ_curr, A_prev.T) / m

    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}

    m = Y.shape
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer.get("activation")

        dA_curr = dA_prev

        A_prev = memory.get("A" + str(layer_idx_prev))
        Z_curr = memory.get("Z" + str(layer_idx_curr))

        W_curr = params_values.get("w" + str(layer_idx_curr))
        b_curr = params_values.get("b" + str(layer_idx_curr))

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr
        )

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    ## enumerate(list, num) -> start with num
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values.get("dW" + str(layer_idx))
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values.get("db" + str(layer_idx))

    return params_values


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if(i % 50 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)

    return  params_values


def one_hot_encod(dataset):
    size = dataset.shape[-1]
    one_hot_shape = (size, 10)
    one_hot_y = np.zeros(one_hot_shape)
    one_hot_y[np.arange(size), dataset] = 1

    return one_hot_y.T


def flatten(dataset):
    size = dataset.shape[0]
    data_shape = (size, 784)
    dataset = dataset.reshape(data_shape)

    return dataset


if __name__ == "__main__":
    ## load dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ## process data
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    y_train = one_hot_encod(y_train)
    y_test = one_hot_encod(y_test)
    print(y_test.shape)
