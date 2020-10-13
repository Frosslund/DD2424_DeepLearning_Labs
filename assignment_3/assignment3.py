import numpy as np
import pickle
import unittest
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


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


class Neural_Network():

    def __init__(self, train_data, val_data, test_data, layers, batch_norm):

        self.x_train = train_data[0]
        self.y_train = train_data[1]
        self.label_train = train_data[2]

        self.x_val = val_data[0]
        self.y_val = val_data[1]
        self.label_val = val_data[2]

        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.label_test = test_data[2]

        self.layers = layers
        self.batch_norm = batch_norm
        self.test_time = False

        if self.batch_norm:
            self.z_reg = []
            self.z_hat = []
            self.mu = []
            self.var = []
            self.alpha = 0.9

        self.w, self.b, self.gamma, self.beta, self.mu_avg, self.var_avg = self.initialize_parameters(
            10)

    def set_test(self):
        self.test_time = True

    def initialize_parameters(self, k):
        """ Creates and returns weight matrix W and bias vector b """

        # np.random.seed(400)
        w = []
        b = []
        gamma = []
        beta = []
        mu_avg = []
        var_avg = []

        for i in range(len(self.layers)+1):
            if i == 0:
                w.append(np.random.normal(
                    0, 0.1, (self.layers[0], self.x_train.shape[0])))
                b.append(np.zeros((self.layers[i], 1)))
                gamma.append(np.ones((self.layers[i], 1)))
                beta.append(np.zeros((self.layers[i], 1)))
                mu_avg.append(np.zeros((self.layers[i], 1)))
                var_avg.append(np.zeros((self.layers[i], 1)))
            elif i == len(self.layers):
                w.append(np.random.normal(
                    0, 0.1, (k, self.layers[i-1])))
                b.append(np.zeros((k, 1)))
                gamma.append(np.ones((k, 1)))
                beta.append(np.zeros((k, 1)))
                mu_avg.append(np.zeros((k, 1)))
                var_avg.append(np.zeros((k, 1)))
            else:
                w.append(np.random.normal(
                    0, 0.1, (self.layers[i], self.layers[i-1])))
                b.append(np.zeros((self.layers[i], 1)))
                gamma.append(np.ones((self.layers[i], 1)))
                beta.append(np.zeros((self.layers[i], 1)))
                mu_avg.append(np.zeros((self.layers[i], 1)))
                var_avg.append(np.zeros((self.layers[i], 1)))

        return w, b, gamma, beta, mu_avg, var_avg

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def relu(self, x):
        """ ReLU activation function """
        x[x < 0] = 0
        return x

    def evaluate_classifier(self, x):
        # forward pass
        """ Computes and returns the softmax matrix """
        if self.batch_norm:
            h, self.mu, self.var, self.z_reg, self.z_hat = [], [], [], [], []
        else:
            h = []

        h.append(x)
        n = x.shape[1]

        for i in range(len(self.layers)):
            if i == 0:
                z = np.matmul(self.w[i], x) + self.b[i]
                if self.batch_norm:
                    z = self.batch_norm_forward(z, i, n)
                h.append(self.relu(z))
            else:
                z = np.matmul(self.w[i], h[i]) + self.b[i]
                if self.batch_norm:
                    z = self.batch_norm_forward(z, i, n)
                h.append(self.relu(z))

        z = np.matmul(self.w[-1], h[-1]) + self.b[-1]
        # ska detta vara här
        """ if self.batch_norm:
            z = self.batch_norm_forward(z, i, n) """
        p = self.softmax(z)

        return h, p

    def batch_norm_forward(self, z, i, n):
        self.z_reg.append(z)

        if self.test_time:
            z = (z - self.mu_avg[i]) / \
                np.sqrt(self.var_avg[i] + np.finfo(np.float64).eps)
        else:
            mu_curr = np.mean(z, axis=1, keepdims=True)
            # print(mu_curr)
            self.mu.append(mu_curr)

            var_curr = np.var(z, axis=1, keepdims=True)
            self.var.append(var_curr)

            self.mu_avg[i] = self.alpha * \
                self.mu_avg[i] + (1-self.alpha) * mu_curr
            self.var_avg[i] = self.alpha * \
                self.var_avg[i] + (1-self.alpha) * var_curr

            z = (z - mu_curr) / np.sqrt(var_curr + np.finfo(np.float64).eps)

        self.z_hat.append(z)

        return np.multiply(self.gamma[i], z) + self.beta[i]

    def batch_norm_backward(self, g, z, mu, var):
        n = g.shape[1]
        sigma_one = np.power(var + np.finfo(np.float64).eps, -0.5)
        sigma_two = np.power(var + np.finfo(np.float64).eps, -1.5)

        g_one = np.multiply(g, sigma_one)
        g_two = np.multiply(g, sigma_two)

        d = z - mu
        c = np.sum(np.multiply(g_two, d), axis=1, keepdims=True)

        g = g_one - (1/n) * np.sum(g_one, axis=1, keepdims=True) - \
            (1/n) * np.multiply(d, c)

        return g

    def compute_cost(self, x, y, lamb):
        """ Computes and returns the cross-entropy loss """
        _, p = self.evaluate_classifier(x)

        tot_sum = 0
        for weight_matrix in self.w:
            tot_sum += np.sum(np.square(weight_matrix))

        loss = (1/x.shape[1]) * -np.sum(y*np.log(p))
        j = loss + lamb * tot_sum

        return loss, j

    def compute_accuracy(self, x, y_labels):
        """ Computes and returns the accuracy of the classifier """
        _, p = self.evaluate_classifier(x)
        predictions = np.argmax(p, axis=0).T
        return np.array(np.where(predictions == np.array(y_labels))).shape[1] / np.size(y_labels)

    def compute_grads_num_slow(self, x, y, w, b, gamma, beta, lamda, eps):
        """ Computation of numerical gradients. Converted from matlab code and provided in the assignment """

        grad_w = [np.zeros(np.array(w).shape) for ww in w]
        grad_b = [np.zeros(np.array(b).shape) for bb in b]
        grad_gamma = [np.ones(gamma.shape) for gamma in gamma]
        grad_beta = [np.zeros(beta.shape) for beta in beta]
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

        for bias in b:
            i = 0
            for sub_bias in bias:
                j = 0
                bias[j] -= eps
                _, c1 = self.compute_cost(x, y, lamda)
                bias[j] += 2*eps
                _, c2 = self.compute_cost(x, y, lamda)
                bias[j] -= eps
                grad_b[i][j] = (c2-c1) / (2*eps)
                j += 1
            i += 1

        for i, g in enumerate(range(gamma)):
            for j in range(len(g)):
                g[j] -= eps
                c1 = self.compute_cost(x, y, lamda)
                g[j] += 2*eps
                c2 = self.compute_cost(x, y, lamda)
                g[j] -= eps
                grad_gamma[i, j] = (c2-c1) / (2*eps)

        for i, b in enumerate(range(len(beta))):
            for j in range(len(b)):
                b[j] -= eps
                c1 = self.compute_cost(x, y, lamda)
                b[j] += 2*eps
                c2 = self.compute_cost(x, y, lamda)
                b[j] -= eps
                grad_beta[i, j] = (c2-c1) / (2*eps)

        for i, weight in enumerate(range(w)):
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    weight[j, k] -= eps
                    c1 = self.compute_cost(x, y, lamda)
                    weight[j, k] += 2*eps
                    c2 = self.compute_cost(x, y, lamda)
                    weight[j, k] -= eps
                    grad_w[i, j, k] = (c2-c1) / (2*eps)

        return np.array(grad_w), np.array(grad_b), np.array(grad_gamma), np.array(grad_beta)

    def compute_gradients(self, x, y, lamb, n_batch, w):
        # backward pass
        """ Analytical computation of gradients """
        h, p = self.evaluate_classifier(x)
        grad_w = []
        grad_b = []
        if self.batch_norm:
            grad_gamma = []
            for gamma in self.gamma:
                grad_gamma.append(np.zeros_like(gamma))
            grad_beta = []
            for beta in self.beta:
                grad_beta.append(np.zeros_like(beta))

        g = -(y-p)

        grad_w.insert(0, (1/n_batch) * np.matmul(g,
                                                 h[-1].T) + (2 * lamb * self.w[-1]))
        grad_b.insert(0, np.array(
            (1/n_batch)*np.matmul(g, np.ones(n_batch))).reshape(np.size(self.w[-1], 0), 1))

        # inget i gamma och beta här?

        g = np.matmul(self.w[-1].T, g)
        h[-1] = np.where(np.array(h[-1]) > 0, 1, 0)
        g = np.multiply(g, h[-1] > 0)
        for i in range(len(self.layers)-1, -1, -1):
            if self.batch_norm:
                grad_gamma[i] = (np.array(
                    (1/n_batch)*np.matmul(np.multiply(g, self.z_hat[i]), np.ones(n_batch)).reshape(np.size(grad_gamma[i], 0), 1)))
                grad_beta[i] = (np.array(
                    (1/n_batch) * np.matmul(g, np.ones(n_batch)).reshape(np.size(grad_beta[i], 0), 1)))

                g = np.multiply(g, self.gamma[i])
                g = self.batch_norm_backward(
                    g, self.z_reg[i], self.mu[i], self.var[i])

            grad_w.insert(0, (1/n_batch) * np.matmul(g,
                                                     h[i].T) + (2 * lamb * self.w[i]))
            grad_b.insert(0, np.array(
                (1/n_batch)*np.matmul(g, np.ones(n_batch))).reshape(np.size(self.w[i], 0), 1))

            if i > 0:
                g = np.matmul(self.w[i].T, g)
                h[i] = np.where(np.array(h[i]) > 0, 1, 0)
                g = np.multiply(g, h[i] > 0)

        """ grad_w.insert(0, (1/n_batch) * np.matmul(g, np.array(x).T) +
                      (2 * lamb * self.w[0]))
        grad_b.insert(0, np.array((1/n_batch)*np.matmul(g, np.ones(n_batch))
                                  ).reshape(np.size(self.w[0], 0), 1)) """

        if self.batch_norm:
            return grad_w, grad_b, grad_gamma, grad_beta
        else:
            return grad_w, grad_b

    def create_batches(self, x, y, n_batch):
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

    def random_search_optimization(self):

        l_min = 0.005
        l_max = 0.012

        lamb = np.random.uniform(l_min, l_max)

        train_cost, train_loss, val_cost, val_loss, train_acc, val_acc, test_acc, n_epochs = self.mini_batch_gd(
            lamb, 20)

        val_acc = self.compute_accuracy(self.x_val, self.label_val)
        test_acc = self.compute_accuracy(self.x_test, self.label_test)

        return [lamb, val_acc, test_acc]

    def mini_batch_gd(self, lamb, n_epochs, computations=True, cyclical_eta=True):
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
            n_s = 5 * np.floor(self.x_train.shape[1] / n_batches)
            #n_s = 800
            t = 0
            l = 0
        else:
            eta = 0.001

        x_batches, y_batches = self.create_batches(
            self.x_train, self.y_train, n_batches)

        if computations:
            loss, cost = self.compute_cost(
                self.x_train, self.y_train, lamb)
            train_cost.append(cost)
            train_loss.append(loss)
            train_acc.append(self.compute_accuracy(
                self.x_train, self.label_train))
            loss, cost = self.compute_cost(
                self.x_val, self.y_val, lamb)
            val_cost.append(cost)
            val_loss.append(loss)
            val_acc.append(self.compute_accuracy(
                self.x_val, self.label_val))

        for epoch in range(n_epochs):
            for batch in range(len(x_batches)):

                if self.batch_norm:
                    grad_w, grad_b, grad_gamma, grad_beta = self.compute_gradients(
                        x_batches[batch], y_batches[batch], lamb, n_batches, self.w)
                else:
                    grad_w, grad_b = self.compute_gradients(
                        x_batches[batch], y_batches[batch], lamb, n_batches, self.w)

                for i in range(len(self.w)):
                    self.w[i] = self.w[i]-eta*grad_w[i]
                for i in range(len(self.b)):
                    self.b[i] = self.b[i]-eta*grad_b[i]
                    if self.batch_norm:
                        self.gamma[i] = self.gamma[i]-eta*grad_gamma[i]
                        self.beta[i] = self.beta[i]-eta*grad_beta[i]

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
                loss, cost = self.compute_cost(
                    self.x_train, self.y_train, lamb)
                train_cost.append(cost)
                print(cost)
                train_loss.append(loss)
                train_acc.append(self.compute_accuracy(
                    self.x_train, self.label_train))
                loss, cost = self.compute_cost(
                    self.x_val, self.y_val, lamb)
                val_cost.append(cost)
                val_loss.append(loss)
                val_acc.append(self.compute_accuracy(
                    self.x_val, self.label_val))

            print("Epoch: " + str(epoch+1) + " completed.")

        self.set_test()
        test_acc = self.compute_accuracy(self.x_test, self.label_test)
        print(test_acc)
        print(val_acc[-1])
        return train_cost, train_loss, val_cost, val_loss, train_acc, val_acc, test_acc, n_epochs


def plot_cost(train_cost, val_cost, epochs):
    """ Plots the cost of training and validation data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train_cost, label="Training Data")
    ax.plot(epochs_label, val_cost, label="Validation Data")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Cost', ylim=(0, 4), xlim=(0, epochs+2))
    ax.grid()

    plt.show()


def plot_loss(train_loss, val_loss, epochs):
    """ Plots the loss of training and validation data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train_loss, label="Training Data")
    ax.plot(epochs_label, val_loss, label="Validation Data")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Loss', ylim=(0, 3), xlim=(0, epochs+2))
    ax.grid()

    plt.show()


def plot_acc(train, val, test, epochs, plot_test=True):
    """ Plots accuracy of training, validation and test data """
    epochs_label = np.arange(0, epochs+1, 1)

    fig, ax = plt.subplots()

    ax.plot(epochs_label, train, label="Training Accuracy")
    ax.plot(epochs_label, val, label="Validation Accuracy")
    if plot_test:
        ax.plot(epochs, test, marker='o', markersize=7, label="Test Accuracy")
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1), xlim=(0, epochs+2))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.show()


def initialization(all=True):
    """ Loads data, initializes model parameters and normalizes data, either all available data or individual batches"""

    if all:
        train_data, val_data, test_data = load_all_batches(5000)

        train_data[0], val_data[0], test_data[0] = normalizing_training_data(
            train_data[0], val_data[0], test_data[0])

        return train_data, val_data, test_data

    else:
        train_x, train_y, train_labels = load_batch('data_batch_1')
        val_x, val_y, val_labels = load_batch('data_batch_2')
        test_x, test_y, test_labels = load_batch('test_batch')

        train_x, val_x, test_x = normalizing_training_data(
            train_x, val_x, test_x)

        return [train_x, train_y, train_labels], [val_x, val_y, val_labels], [test_x, test_y, test_labels]


def gradient_check(grad_w_analyt, grad_w_num, grad_b_analyt, grad_b_num, grad_gamma_analyt, grad_gamma_num, grad_beta_analyt, grad_beta_num, eps):
    """ Function providing preprocessing for calculating relative error between analytical and numerical gradients """
    checks_w = []
    checks_b = []
    checks_gamma = []
    checks_beta = []

    for k in range(len(grad_w_analyt)):
        for d in range(len(grad_w_analyt[k])):
            checks_w.append(np.abs(grad_w_analyt[k][d] - grad_w_num[k][d]) / max(
                eps, (np.abs(grad_w_analyt[k][d]) + np.abs(grad_w_num[k][d]))))

    for k in range(len(grad_b_analyt)):
        checks_b.append(np.abs(grad_b_analyt[k] - grad_b_num[k]) / max(
            eps, (np.abs(grad_b_analyt[k]) + np.abs(grad_b_num[k]))))

    for k in range(len(grad_gamma_analyt)):
        checks_b.append(np.abs(grad_gamma_analyt[k] - grad_gamma_num[k]) / max(
            eps, (np.abs(grad_gamma_analyt[k]) + np.abs(grad_gamma_num[k]))))

    for k in range(len(grad_beta_analyt)):
        checks_b.append(np.abs(grad_beta_analyt[k] - grad_beta_num[k]) / max(
            eps, (np.abs(grad_beta_analyt[k]) + np.abs(grad_beta_num[k]))))

    return checks_w, checks_b, checks_gamma, checks_beta


def gradientCheck(self, gradW_a, gradW_n, gradB_a, gradB_n, eps):
    """ computes the relative error between analytical and numerical gradient calcs """

    def check(grad_a, grad_n, eps):
        diff = np.absolute(np.subtract(grad_a, grad_n))
        thresh = np.full(diff.shape, eps)
        summ = np.add(np.absolute(grad_a), np.absolute(grad_n))
        denom = np.maximum(thresh, summ)
        return np.divide(diff, denom)

    resW = []
    resB = []
    for i in range(len(gradW_a)):
        resW.append(check(gradW_a[i], gradW_n[i], eps))
        resB.append(check(gradB_a[i], gradB_n[i], eps))
    return resW, resB


def main(search=False, all=True):

    train_data, val_data, test_data = initialization()

    if search:
        results = []
        for i in range(10):
            nn = Neural_Network(train_data, val_data,
                                test_data, [50, 50], True)
            result = nn.random_search_optimization()
            results.append(result)
            print("Network completion")
        for result in results:
            print(result)
    else:
        nn = Neural_Network(train_data, val_data, test_data, [
                            50, 30, 20, 20, 10, 10, 10, 10], False)
        train_cost, train_loss, val_cost, val_loss, train_acc, val_acc, test_acc, n_epochs = nn.mini_batch_gd(
            0.010530052307840956, 20)

    #test_acc = nn.compute_accuracy(self.x_test, self.label_test)

    plot_cost(train_cost, val_cost, n_epochs)
    plot_loss(train_loss, val_loss, n_epochs)
    plot_acc(train_acc, val_acc, test_acc, n_epochs)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.train_data, self.val_data, self.test_data = initialization()
        nn = Neural_Network(self.train_data, self.val_data,
                            self.test_data, [50, 50], True)

        self.w, self.b, self.gamma, self.beta, _, _ = nn.initialize_parameters(
            10)

        self.lamb = 0

        self.grad_w_num, self.grad_b_num, self.grad_gamma_num, self.grad_beta_num = nn.compute_grads_num_slow(
            self.train_data[0][:50][:1], self.train_data[1][:][:1], self.w[:][:50], self.b, self.gamma, self.beta, self.lamb, 10**-5)

        self.grad_w_analyt, self.grad_b_analyt, self.grad_gamma_analyt, self.grad_beta_analyt = nn.compute_gradients(
            np.array(self.train_data[0][:50, :1]), np.array(self.train_data[1][:, :1]), self.lamb, 1, np.array(self.w[:, :50]))

        self.checks_w, self.checks_b, self.checks_gamma, self.checks_beta = gradient_check(
            self.grad_w_analyt, self.grad_w_num, self.grad_b_analyt, self.grad_b_num, self.grad_gamma_analyt, self.grad_gamma_num, self.grad_beta_analyt, self.grad_beta_num, 10**-5)

    def test_gradient_mean(self):

        self.assertAlmostEqual(self.grad_w_num.mean(),
                               self.grad_w_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_b_num.mean(),
                               self.grad_b_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_gamma_num.mean(),
                               self.grad_gamma_analyt.mean(), places=9)
        self.assertAlmostEqual(self.grad_beta_num.mean(),
                               self.grad_beta_analyt.mean(), places=9)

    def test_rel_err(self):
        self.assertLessEqual(np.max(self.checks_w), 10**-6)
        self.assertLessEqual(np.max(self.checks_b), 10**-5)
        self.assertLessEqual(np.max(self.checks_gamma), 10**-6)
        self.assertLessEqual(np.max(self.checks_beta), 10**-6)


if __name__ == "__main__":
    main()
