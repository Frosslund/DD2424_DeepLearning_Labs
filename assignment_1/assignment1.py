import numpy as np
import pickle
import unittest
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_batch(filename):
    """ Function to load any of the provided batches of images """
    with open('./datasets/'+filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = np.array(data.get(b'data'), dtype=float).T
    y_labels = np.array([data.get(b'labels')])
    y_one_hot = np.zeros((10, x.shape[1]))

    y_one_hot[y_labels, np.arange(y_labels.size)] = 1

    return x, y_one_hot, y_labels


def normalizing_training_data(training_data, validation_data, test_data):
    """ Normalizes training, validation and test data to zero mean and .01 standard deviation """
    mean_train = np.array([np.mean(training_data, 1)]).T
    std_train = np.array([np.std(training_data, 1)]).T

    training_data = (training_data - mean_train) / std_train
    validation_data = (validation_data - mean_train) / std_train
    test_data = (test_data - mean_train) / std_train
    return training_data, validation_data, test_data


def initialize_parameters(k, d):
    """ Creates and returns weight matrix W and bias vector b """
    mu, sigma = 0, 0.01
    np.random.seed(400)

    w = np.random.normal(mu, sigma, (k, d))
    b = np.random.normal(mu, sigma, (k, 1))
    return w, b


def evaluate_classifier(x, w, b):
    """ Computes and returns the softmax matrix """
    return softmax(np.matmul(w, x) + b)


def compute_cost(x, y, w, b, lamb):
    """ Computes and returns the cross-entropy loss """
    p = evaluate_classifier(x, w, b)
    j = 1/np.size(x, 1) * -np.sum(y*np.log(p)) + \
        (lamb*(np.sum(np.square(w))))
    return j


def compute_accuracy(x, w, b, y_labels):
    """ Computes and returns the accuracy of the classifier """
    p = evaluate_classifier(x, w, b)
    predictions = np.argmax(p, axis=0)
    return np.array(np.where(predictions == np.array(y_labels))).shape[1] / np.size(y_labels)


def compute_grads_num(x, y, w, b, lamda, h):
    """ Computation of numerical gradients. Converted from matlab code and provided in the assignment """
    no = w.shape[0]
    d = x.shape[0]

    grad_w = np.zeros(w.shape)
    grad_b = np.zeros(b.shape)

    c = compute_cost(x, y, w, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(x, y, w, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_try = np.array(w)
            w_try[i, j] += h
            c2 = compute_cost(x, y, w_try, b, lamda)
            grad_w[i, j] = (c2-c) / h

    return grad_w, grad_b


def compute_grads_num_slow(x, y, w, b, lamda, h):
    """ Computation of numerical gradients. Converted from matlab code and provided in the assignment """

    grad_w = np.zeros(w.shape)
    grad_b = np.zeros(b.shape)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = compute_cost(x, y, w, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(x, y, w, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_try = np.array(w)
            w_try[i, j] -= h
            c1 = compute_cost(x, y, w_try, b, lamda)

            w_try = np.array(w)
            w_try[i, j] += h

            c2 = compute_cost(x, y, w_try, b, lamda)

            grad_w[i, j] = (c2-c1) / (2*h)

    return grad_w, grad_b


def compute_gradients(x, y, b, w, lamb, n_batch):
    """ Analytical computation of gradients """

    p = evaluate_classifier(x, w, b)

    g = -(y-p)

    grad_w = (1/n_batch) * np.matmul(g, np.array(x).T) + (2 * lamb * w)
    grad_b = np.array((1/n_batch)*np.matmul(g, np.ones(n_batch))
                      ).reshape(np.size(w, 0), 1)
    return grad_w, grad_b


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
    epochs_label = np.arange(1, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train_cost, label="Training Data")
    ax.plot(epochs_label, val_cost, label="Validation Data")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Cost')
    ax.grid()

    plt.show()


def plot_acc(train, val, test, epochs):
    """ Plots accuracy of training, validation and test data """
    epochs_label = np.arange(1, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train, label="Training Accuracy")
    ax.plot(epochs_label, val, label="Validation Accuracy")
    ax.plot(epochs, test, marker='o', markersize=7, label="Test Accuracy")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Accuracy')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.show()


def plot_imgs(w):
    """ Display the image for each label in W. Converted from matlab code and provided in the assignment """
    w = np.array(w)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = w[i+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("Class "+str(5*i+j+1))
            ax[i][j].axis('off')
    plt.show()


def mini_batch_gd(train_data, val_data, w, b, eta_decay=False):
    """ Performs mini-batch gradient descent with chosen parameter settings """

    n_batches = 100
    eta = 0.001
    n_epochs = 40
    lamb = 1.0
    train_res = []
    train_acc = []
    val_res = []
    val_acc = []

    x_batches, y_batches = create_batches(
        train_data[0], train_data[1], n_batches)

    for epoch in range(n_epochs):

        for batch in range(len(x_batches)):

            grad_w, grad_b = compute_gradients(
                x_batches[batch], y_batches[batch], b, w, lamb, n_batches)
            w = w-(eta*grad_w)
            b = b-(eta*grad_b)

        print(compute_cost(
            train_data[0], train_data[1], w, b, lamb))
        train_res.append(compute_cost(
            train_data[0], train_data[1], w, b, lamb))
        train_acc.append(compute_accuracy(train_data[0], w, b, train_data[2]))
        val_res.append(compute_cost(val_data[0], val_data[1], w, b, lamb))
        val_acc.append(compute_accuracy(val_data[0], w, b, val_data[2]))
        print("Epoch: " + str(epoch))

    return train_res, val_res, train_acc, val_acc, w, b, n_epochs


def main():

    train_x, train_y, train_labels = load_batch('data_batch_1')
    val_x, val_y, val_labels = load_batch('data_batch_2')
    test_x, _, test_labels = load_batch('test_batch')

    train_x, val_x, test_x = normalizing_training_data(
        train_x, val_x, test_x)

    w, b = initialize_parameters(10, train_x.shape[0])
    train_res, val_res, train_acc, val_acc, w, b, n_epochs = mini_batch_gd(
        [train_x, train_y, train_labels], [val_x, val_y, val_labels], w, b)

    test_acc = compute_accuracy(test_x, w, b, test_labels)

    plot_cost(train_res, val_res, n_epochs)
    plot_acc(train_acc, val_acc, test_acc, n_epochs)
    plot_imgs(w)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.train_data = load_batch('data_batch_1')
        self.w, self.b = initialize_parameters(10, self.train_data[0].shape[0])
        self.lamb = 0

        self.grad_w_num, self.grad_b_num = compute_grads_num_slow(
            self.train_data[0][:, :1], self.train_data[1][:, :1], self.w, self.b, self.lamb, 10**-6)
        self.grad_w_analyt, self.grad_b_analyt = compute_gradients(
            self.train_data[0][:, :1], self.train_data[1][:, :1], self.b, self.w, self.lamb, 1)

        self.checks_w, self.checks_b = gradient_check(
            self.grad_w_analyt, self.grad_w_num, self.grad_b_analyt, self.grad_b_num, 10**-6)

    def test_gradient_mean(self):

        self.assertAlmostEqual(self.grad_w_num.mean(),
                               self.grad_w_analyt.mean(), places=7)
        self.assertAlmostEqual(self.grad_b_num.mean(),
                               self.grad_b_analyt.mean(), places=7)

    def test_rel_err(self):

        self.assertLessEqual(np.max(self.checks_w), 10**-5)
        self.assertLessEqual(np.max(self.checks_b), 10**-6)


if __name__ == "__main__":
    main()
