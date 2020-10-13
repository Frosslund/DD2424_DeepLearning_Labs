import numpy as np
import pickle
import unittest
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    """ ReLU activation function """
    x[x < 0] = 0
    return x


def load_batch(filename):
    """ Function to load any of the provided batches of images """
    with open('./datasets/'+filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = np.array(data.get(b'data'), dtype=float).T
    y_labels = np.array([data.get(b'labels')])
    y_one_hot = np.zeros((10, x.shape[1]))

    y_one_hot[y_labels, np.arange(y_labels.size)] = 1

    return x, y_one_hot, y_labels


def load_all_batches(num):
    """ Function to load all of the provided batches of images """
    train_data = []
    val_data = []
    test_data = []
    for i in range(1, 6):
        x, y, labels = load_batch('data_batch_'+str(i))

        if i == 1:
            train_data.append(x)
            train_data.append(y)
            train_data.append(labels)
        elif i == 5:
            train_data[0] = np.concatenate(
                (train_data[0], x[:, :num]), axis=1)
            train_data[1] = np.concatenate(
                (train_data[1], y[:, :num]), axis=1)
            train_data[2] = np.concatenate(
                (train_data[2], labels[:, :num]), axis=1)
            val_data.append(x[:, num:])
            val_data.append(y[:, num:])
            val_data.append(labels[:, num:])
        else:
            train_data[0] = np.concatenate((train_data[0], x), axis=1)
            train_data[1] = np.concatenate((train_data[1], y), axis=1)
            train_data[2] = np.concatenate((train_data[2], labels), axis=1)

    x, y, labels = load_batch('test_batch')
    test_data.append(x)
    test_data.append(y)
    test_data.append(labels)

    return train_data, val_data, test_data


def normalizing_training_data(training_data, validation_data, test_data):
    """ Normalizes training, validation and test data to zero mean and .01 standard deviation """
    mean_train = np.array([np.mean(training_data, 1)]).T
    std_train = np.array([np.std(training_data, 1)]).T

    training_data = (training_data - mean_train) / std_train
    validation_data = (validation_data - mean_train) / std_train
    test_data = (test_data - mean_train) / std_train
    return training_data, validation_data, test_data


def initialize_parameters(k, d, m=50):
    """ Creates and returns weight matrix W and bias vector b """
    # w1 mxd, w2 kxm, b1 mx1, b2 kx1
    np.random.seed(400)
    w1 = np.random.normal(
        0,
        1/np.sqrt(d),
        (m, d))
    w2 = np.random.normal(
        0,
        1/np.sqrt(m),
        (k, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))
    return w1, w2, b1, b2


def evaluate_classifier(x, w1, w2, b1, b2):
    # forward pass
    """ Computes and returns the softmax matrix """
    s1 = np.matmul(w1, x) + b1
    h = relu(s1)
    s2 = np.matmul(w2, h) + b2
    p = softmax(s2)
    return h, p


def compute_cost(x, y, w1, w2, b1, b2, lamb):
    """ Computes and returns the cross-entropy loss """
    _, p = evaluate_classifier(x, w1, w2, b1, b2)
    loss = (1/x.shape[1]) * -np.sum(y*np.log(p))
    j = loss + lamb * \
        (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return loss, j


def compute_accuracy(x, w1, w2, b1, b2, y_labels):
    """ Computes and returns the accuracy of the classifier """
    _, p = evaluate_classifier(x, w1, w2, b1, b2)
    predictions = np.argmax(p, axis=0).T
    return np.array(np.where(predictions == np.array(y_labels))).shape[1] / np.size(y_labels)


def compute_grads_num_slow(x, y, w1, w2, b1, b2, lamda, eps):
    """ Computation of numerical gradients. Converted from matlab code and provided in the assignment """

    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= eps
        c1 = compute_cost(x, y, w1, w2, b1_try, b2, lamda)

        b1_try = np.array(b1)
        b1_try[i] += eps

        c2 = compute_cost(x, y, w1, w2, b1_try, b2, lamda)
        grad_b1[i] = (c2-c1) / (2*eps)

    for i in range(len(b2)):

        b2_try = np.array(b2)
        b2_try[i] -= eps
        c1 = compute_cost(x, y, w1, w2, b1, b2_try, lamda)

        b2_try = np.array(b2)
        b2_try[i] += eps

        c2 = compute_cost(x, y, w1, w2, b1, b2_try, lamda)
        grad_b2[i] = (c2-c1) / (2*eps)

    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w1_try = np.array(w1)
            w1_try[i, j] -= eps
            c1 = compute_cost(x, y, w1_try, w2, b1, b2, lamda)

            w1_try = np.array(w1)
            w1_try[i, j] += eps

            c2 = compute_cost(x, y, w1_try, w2, b1, b2, lamda)

            grad_w1[i, j] = (c2-c1) / (2*eps)

    for i in range(w2.shape[0]):
        for j in range(w2.shape[1]):
            w2_try = np.array(w2)
            w2_try[i, j] -= eps
            c1 = compute_cost(x, y, w1, w2_try, b1, b2, lamda)

            w2_try = np.array(w2)
            w2_try[i, j] += eps

            c2 = compute_cost(x, y, w1, w2_try, b1, b2, lamda)

            grad_w2[i, j] = (c2-c1) / (2*eps)

    return grad_w1, grad_w2, grad_b1, grad_b2


def compute_gradients(x, y, w1, w2, b1, b2, lamb, n_batch):
    # backward pass
    """ Analytical computation of gradients """
    h, p = evaluate_classifier(x, w1, w2, b1, b2)

    g = -(y-p)

    grad_w2 = (1/n_batch) * np.matmul(g, np.array(h).T) + (2 * lamb * w2)
    grad_b2 = np.array((1/n_batch)*np.matmul(g, np.ones(n_batch))
                       ).reshape(np.size(w2, 0), 1)

    g = np.matmul(w2.T, g)
    h = np.where(h > 0, 1, 0)
    g = np.multiply(g, h > 0)

    grad_w1 = (1/n_batch) * np.matmul(g, np.array(x).T) + (2 * lamb * w1)
    grad_b1 = np.array((1/n_batch)*np.matmul(g, np.ones(n_batch))
                       ).reshape(np.size(w1, 0), 1)

    return grad_w1, grad_w2, grad_b1, grad_b2


def create_batches(x, y, n_batch):
    """ Segmentation of training data in mini batches """
    x_batches = []
    y_batches = []
    for j in range(int(x.shape[1]/n_batch)):
        n = j*n_batch
        x_batch = x[:, n:n+n_batch]
        y_batch = y[:, n:n+n_batch]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches


def gradient_check(grad_w_analyt, grad_w_num, grad_b_analyt, grad_b_num, eps):
    """ Function providing preprocessing for calculating relative error between analytical and numerical gradients """
    checks_w = []
    checks_b = []
    for k in range(len(grad_w_analyt)):
        for d in range(len(grad_w_analyt[k])):
            checks_w.append(np.abs(grad_w_analyt[k][d] - grad_w_num[k][d]) / max(
                eps, (np.abs(grad_w_analyt[k][d]) + np.abs(grad_w_num[k][d]))))

    for k in range(len(grad_b_analyt)):
        checks_b.append(np.abs(grad_b_analyt[k] - grad_b_num[k]) / max(
            eps, (np.abs(grad_b_analyt[k]) + np.abs(grad_b_num[k]))))
    return checks_w, checks_b


def plot_cost(train_cost, val_cost, epochs):
    """ Plots the cost of training and validation data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train_cost, label="Training Data")
    ax.plot(epochs_label, val_cost, label="Validation Data")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Cost', ylim=(0, 4), xlim=(0, 22))
    ax.grid()

    plt.show()


def plot_loss(train_loss, val_loss, epochs):
    """ Plots the loss of training and validation data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train_loss, label="Training Data")
    ax.plot(epochs_label, val_loss, label="Validation Data")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Loss', ylim=(0, 3), xlim=(0, 22))
    ax.grid()

    plt.show()


def plot_acc(train, val, test, epochs, plot_test=False):
    """ Plots accuracy of training, validation and test data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train, label="Training Accuracy")
    ax.plot(epochs_label, val, label="Validation Accuracy")
    if plot_test:
        ax.plot(epochs, test, marker='o', markersize=7, label="Test Accuracy")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1), xlim=(0, 22))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.show()


def mini_batch_gd(train_data, val_data, w1, w2, b1, b2, lamb, n_epochs, computations=True, cyclical_eta=True):
    """ Performs mini-batch gradient descent with chosen parameter settings """

    n_batches = 100
    eta_min = 10**-5
    eta_max = 10**-1
    train_cost = []
    train_loss = []
    train_acc = []
    val_cost = []
    val_loss = []
    val_acc = []

    if cyclical_eta:
        eta = eta_min
        n_s = 2 * np.floor(train_data[0].shape[1] / n_batches)
        t = 0
        l = 0
    else:
        eta = 0.001

    x_batches, y_batches = create_batches(
        train_data[0], train_data[1], n_batches)

    if computations:
        loss, cost = compute_cost(
            train_data[0], train_data[1], w1, w2, b1, b2, lamb)
        train_cost.append(cost)
        train_loss.append(loss)
        train_acc.append(compute_accuracy(
            train_data[0], w1, w2, b1, b2, train_data[2]))
        loss, cost = compute_cost(
            val_data[0], val_data[1], w1, w2, b1, b2, lamb)
        val_cost.append(cost)
        val_loss.append(loss)
        val_acc.append(compute_accuracy(
            val_data[0], w1, w2, b1, b2, val_data[2]))

    for epoch in range(n_epochs):
        for batch in range(len(x_batches)):

            grad_w1, grad_w2, grad_b1, grad_b2 = compute_gradients(
                x_batches[batch], y_batches[batch], w1, w2, b1, b2, lamb, n_batches)
            w1 = w1-eta*grad_w1
            w2 = w2-eta*grad_w2
            b1 = b1-eta*grad_b1
            b2 = b2-eta*grad_b2

            if cyclical_eta:
                if (2*l*n_s <= t <= (2*l + 1)*n_s):
                    eta = eta_min + ((t-2*l*n_s) / n_s)*(eta_max-eta_min)
                else:
                    eta = eta_max - ((t - ((2*l)+1)*n_s) /
                                     n_s)*(eta_max-eta_min)

                t += 1
                if ((t % (2*n_s)) == 0):
                    l += 1

        if computations:
            loss, cost = compute_cost(
                train_data[0], train_data[1], w1, w2, b1, b2, lamb)
            train_cost.append(cost)
            print(cost)
            train_loss.append(loss)
            train_acc.append(compute_accuracy(
                train_data[0], w1, w2, b1, b2, train_data[2]))
            loss, cost = compute_cost(
                val_data[0], val_data[1], w1, w2, b1, b2, lamb)
            val_cost.append(cost)
            val_loss.append(loss)
            val_acc.append(compute_accuracy(
                val_data[0], w1, w2, b1, b2, val_data[2]))

        print("Epoch: " + str(epoch+1) + " completed.")

    return train_cost, train_loss, val_cost, val_loss, train_acc, val_acc, w1, w2, b1, b2, n_epochs


def initialization(all=False):
    """ Loads data, initializes model parameters and normalizes data, either all available data or individual batches"""

    if all:
        train_data, val_data, test_data = load_all_batches(1000)

        train_data[0], val_data[0], test_data[0] = normalizing_training_data(
            train_data[0], val_data[0], test_data[0])

        w1, w2, b1, b2 = initialize_parameters(10, train_data[0].shape[0])

        return train_data, val_data, test_data, w1, w2, b1, b2

    else:
        train_x, train_y, train_labels = load_batch('data_batch_1')
        val_x, val_y, val_labels = load_batch('data_batch_2')
        test_x, test_y, test_labels = load_batch('test_batch')

        train_x, val_x, test_x = normalizing_training_data(
            train_x, val_x, test_x)

        w1, w2, b1, b2 = initialize_parameters(10, train_x.shape[0])

        return [train_x, train_y, train_labels], [val_x, val_y, val_labels], [test_x, test_y, test_labels], w1, w2, b1, b2


def main(all=True):

    train_data, val_data, test_data, w1, w2, b1, b2 = initialization()

    #random_search_optimization(train_data, val_data, test_data, w1, w2, b1, b2)

    train_cost, train_loss, val_cost, val_loss, train_acc, val_acc, w1, w2, b1, b2, n_epochs = mini_batch_gd(
        train_data, val_data, w1, w2, b1, b2, 0, 20)

    test_acc = compute_accuracy(test_data[0], w1, w2, b1, b2, test_data[2])

    plot_cost(train_cost, val_cost, n_epochs)
    plot_loss(train_loss, val_loss, n_epochs)
    plot_acc(train_acc, val_acc, test_acc, n_epochs)


def random_search_optimization(train_data, val_data, test_data, w1, w2, b1, b2, num=10):
    """ Function to do a random hyperparameter optimization, either through a coarse search or a fine search """

    l_min = 0.001
    l_max = 0.005

    results = []

    for i in range(num):
        lamb = np.random.uniform(l_min, l_max)
        n_epochs = 16

        _, _, _, _, _, _, w1, w2, b1, b2, _ = mini_batch_gd(
            train_data, val_data, w1, w2, b1, b2, lamb, n_epochs)

        _, curr_cost = compute_cost(
            val_data[0], val_data[1], w1, w2, b1, b2, lamb)
        curr_acc = compute_accuracy(val_data[0], w1, w2, b1, b2, val_data[2])
        test_acc = compute_accuracy(test_data[0], w1, w2, b1, b2, test_data[2])

        results.append([lamb, curr_cost, curr_acc, test_acc])

    for result in results:
        print(result)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.train_data = load_batch('data_batch_1')
        self.w1, self.w2, self.b1, self.b2 = initialize_parameters(
            10, self.train_data[0].shape[0])
        self.lamb = 0
        self.grad_w1_num, self.grad_w2_num, self.grad_b1_num, self.grad_b2_num = compute_grads_num_slow(
            self.train_data[0][:50, :1], self.train_data[1][:, :1], self.w1[:, :50], self.w2[:, :50], self.b1, self.b2, self.lamb, 10**-5)
        self.grad_w1_analyt, self.grad_w2_analyt, self.grad_b1_analyt, self.grad_b2_analyt = compute_gradients(
            self.train_data[0][:50, :1], self.train_data[1][:, :1], self.w1[:, :50], self.w2[:, :50], self.b1, self.b2, self.lamb, 1)

        self.checks_w1, self.checks_b1 = gradient_check(
            self.grad_w1_analyt, self.grad_w1_num, self.grad_b1_analyt, self.grad_b1_num, 10**-5)

        self.checks_w2, self.checks_b2 = gradient_check(
            self.grad_w2_analyt, self.grad_w2_num, self.grad_b2_analyt, self.grad_b2_num, 10**-5)

    def test_gradient_mean(self):

        self.assertAlmostEqual(self.grad_w1_num.mean(),
                               self.grad_w1_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_w2_num.mean(),
                               self.grad_w2_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_b1_num.mean(),
                               self.grad_b1_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_b2_num.mean(),
                               self.grad_b2_analyt.mean(), places=9)

    def test_rel_err(self):
        self.assertLessEqual(np.max(self.checks_w1), 10**-6)
        self.assertLessEqual(np.max(self.checks_w2), 10**-5)
        self.assertLessEqual(np.max(self.checks_b1), 10**-6)
        self.assertLessEqual(np.max(self.checks_b2), 10**-6)


if __name__ == "__main__":
    main()
