import numpy as np
from matplotlib import pyplot as plt


def read_data():
    """ 
        Function to read in data and build a dictionary of the various required properties of the data.
    """
    book_fname = 'goblet_book.txt'
    fid = open(book_fname, 'r').read()

    book_data = fid
    book_chars = list(set(fid))
    k = len(book_chars)

    # char_to_ind, chars as keys
    # ind_to_char, ints as keys
    char_to_ind = {}
    ind_to_char = {}

    for i, val in enumerate(book_chars):
        char_to_ind[val] = i
        ind_to_char[i] = val

    return {'book_data': book_data, 'book_chars': book_chars, 'k': k, 'char_to_ind': char_to_ind, 'ind_to_char': ind_to_char}


class RNN():

    def __init__(self, data, m, eta, seq_length):

        self.data = data
        self.m = m
        self.eta = eta
        self.seq_length = seq_length

        self.b, self.c, self.W, self.U, self.V = self.initialize_hyperparameters(
            m, self.data['k'])

        self.grads = {'W': np.zeros_like(self.W), 'V': np.zeros_like(self.V), 'U': np.zeros_like(
            self.U), 'b': np.zeros_like(self.b), 'c': np.zeros_like(self.c)}

        self.m_values = {'W': np.zeros_like(self.W), 'V': np.zeros_like(self.V), 'U': np.zeros_like(
            self.U), 'b': np.zeros_like(self.b), 'c': np.zeros_like(self.c)}

    def reset_grads(self):
        """ 
            Resets the parameter gradients.
        """
        self.grads = {'W': np.zeros_like(self.W), 'V': np.zeros_like(self.V), 'U': np.zeros_like(
            self.U), 'b': np.zeros_like(self.b), 'c': np.zeros_like(self.c)}

    def initialize_hyperparameters(self, m, k):
        """
            Initializes the model parameters as stated in the assignment. 
        """

        sigma = 0.01

        b = np.zeros((m, 1))
        c = np.zeros((k, 1))

        W = np.random.normal(0, sigma, size=(m, m))
        U = np.random.normal(0, sigma, size=(m, k))
        V = np.random.normal(0, sigma, size=(k, m))

        return b, c, W, U, V

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def tanh(self, x):
        return np.tanh(x)

    def synthesize_chars(self, h0, x0, y, n):
        """ 
            Synthesizes sequence of characters of length n.
        """

        synthesized_text = []
        x = x0
        h = h0

        for i in range(n):
            a, h, o, p = self.evaluate_classifier(h, x, y, i, True)

            cp = np.cumsum(p, axis=0)
            a = np.random.rand()
            ixs = np.nonzero(cp - a > 0)
            ii = ixs[0][0]
            x = np.zeros((self.data["k"], 1))
            x[ii][0] = 1
            synthesized_text.append(x)

        return synthesized_text

    def evaluate_classifier(self, h, x, y, i, synth=False):
        """ 
            Forward pass of the learning process.
        """

        at = np.matmul(self.W, h) + np.matmul(self.U, x) + self.b
        ht = self.tanh(at)

        ot = np.matmul(self.V, ht) + self.c
        pt = self.softmax(ot)

        if not synth:
            loss = -np.log(np.dot(y[:, i].T, pt))
            return loss, at, ht, ot, pt
        else:
            return at, ht, ot, pt

    def compute_gradients(self, x, y, h):
        """ 
            General function for analytical computation of parameter gradients. Encapsulates both forward and backward pass.
        """

        #x_list = {}
        hidden_list = {}
        prod_list = {}

        hidden_list[-1] = np.copy(h)
        loss_sum = 0

        self.reset_grads()

        # forward-pass
        for i in range(x.shape[1]):
            x_t = x[:, i].reshape(x.shape[0], 1)

            loss, a, hidden_list[i], o, prod_list[i] = self.evaluate_classifier(
                hidden_list[i-1], x_t, y, i)

            loss_sum += loss

        # backward-pass
        grad_a = np.zeros_like(a)
        for i in reversed(range(x.shape[1])):
            x_t = x[:, i].reshape(x.shape[0], 1)
            y_t = y[:, i].reshape(y.shape[0], 1)
            grad_o = -(y_t-prod_list[i]).T

            g = grad_o
            self.grads["c"] += g.T

            self.grads["V"] = np.matmul(g.T, hidden_list[i].T)

            grad_h = np.dot(self.V.T, g.T) + np.dot(self.W.T, grad_a)
            grad_a = grad_h * (1 - (hidden_list[i] ** 2))

            g = grad_a
            self.grads["b"] += g

            self.grads["W"] = np.matmul(g, hidden_list[i-1].T)
            self.grads["U"] = np.matmul(g, x_t.T)

        # clipping with np.clip
        self.grads["W"] = np.clip(self.grads["W"], -5, 5)
        self.grads["V"] = np.clip(self.grads["V"], -5, 5)
        self.grads["U"] = np.clip(self.grads["U"], -5, 5)
        self.grads["b"] = np.clip(self.grads["b"], -5, 5)
        self.grads["c"] = np.clip(self.grads["c"], -5, 5)

        return loss_sum, hidden_list[-1]

    def compute_gradients_analyt(self, x, y, h):
        """ 
            Analytical computation of gradients, only utilized in gradient comparison check. 
        """

        #x_list = {}
        hidden_list = {}
        prod_list = {}

        hidden_list[-1] = np.copy(h)
        loss_sum = 0

        self.reset_grads()

        # forward-pass
        for i in range(x.shape[1]):
            x_t = x[:, i].reshape(x.shape[0], 1)

            loss, a, hidden_list[i], o, prod_list[i] = self.evaluate_classifier(
                hidden_list[i-1], x_t, y, i)

            loss_sum += loss

        # backward-pass
        grad_a = np.zeros_like(a)
        for i in reversed(range(x.shape[1])):
            x_t = x[:, i].reshape(x.shape[0], 1)
            y_t = y[:, i].reshape(y.shape[0], 1)
            grad_o = -(y_t-prod_list[i]).T

            g = grad_o
            self.grads["c"] += g.T

            self.grads["V"] = np.matmul(g.T, hidden_list[i].T)

            grad_h = np.dot(self.V.T, g.T) + np.dot(self.W.T, grad_a)
            grad_a = grad_h * (1 - (hidden_list[i] ** 2))

            g = grad_a
            self.grads["b"] += g

            self.grads["W"] = np.matmul(g, hidden_list[i-1].T)
            self.grads["U"] = np.matmul(g, x_t.T)

        return self.grads["W"], self.grads["V"], self.grads["U"], self.grads["b"], self.grads["c"]

    def evaluate_classifier_num(self, h, x, y):
        """ 
            Modified forward pass utilized in gradient comparison check.
        """
        P = {}
        H = {}
        H[-1] = h
        loss = 0
        for t in range(x.shape[1]):
            Xt = x[:, t].reshape(x.shape[0], 1)
            at = np.dot(self.U, Xt) + np.dot(self.W, H[t-1]) + self.b
            H[t] = np.tanh(at)
            ot = np.dot(self.V, H[t]) + self.c
            P[t] = np.exp(ot) / np.sum(np.exp(ot))
            loss += -np.log(np.dot(y[:, t].T, P[t]))
        return loss, 0, H, 0, P

    def compute_gradients_num(self, x, y, h0):
        """ 
            Numerical computation of parameter gradients. 
        """
        h = 1e-4
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)
        du = np.zeros_like(self.U)
        dw = np.zeros_like(self.W)
        dv = np.zeros_like(self.V)
        for i in range(len(self.b)):
            b = self.b
            temp = b
            temp[i] -= h
            self.b[i] = temp[i]
            l1, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
            temp = b
            temp[i] += h
            self.b[i] = temp[i]
            l2, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
            self.b = b
            db[i] = (np.sum(l2) - np.sum(l1)) / (h)
        for i in range(len(self.c)):
            c = self.c
            temp = c
            temp[i] -= h
            self.c[i] = temp[i]
            l1, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
            temp = c
            temp[i] += h
            self.c[i] = temp[i]
            l2, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
            self.c = c
            dc[i] = (np.sum(l2) - np.sum(l1)) / (h)
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                u = self.U
                temp = u
                temp[i][j] -= h
                self.U[i][j] = temp[i][j]
                l1, _, H, _, P = self.evaluate_classifier(h0, x, y, i)
                temp = u
                temp[i][j] += h
                self.U[i][j] = temp[i][j]
                l2, _, H, _, P = self.evaluate_classifier(h0, x, y, i)
                self.U = u
                du[i][j] = (np.sum(l2) - np.sum(l1)) / (h)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                w = self.W
                temp = w
                temp[i][j] -= h
                self.W[i][j] = temp[i][j]
                l1, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
                temp = w
                temp[i][j] += h
                self.W[i][j] = temp[i][j]
                l2, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
                self.W = w
                dw[i][j] = (np.sum(l2) - np.sum(l1)) / (h)
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                v = self.V
                temp = v
                temp[i][j] -= h
                self.V[i][j] = temp[i][j]
                l1, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
                temp = v
                temp[i][j] += h
                self.V[i][j] = temp[i][j]
                l2, _, H, _, P = self.evaluate_classifier_num(h0, x, y)
                self.V = v
                dv[i][j] = (np.sum(l2) - np.sum(l1)) / (h)
        return dw, dv, du, db, dc

    def create_one_hot(self, data):
        """ 
            Function to create a one-hot-encoding from a given set of character data.
        """
        one_hot = np.zeros((self.data["k"], len(data)))

        for b in range(len(data)):
            one_hot[self.data["char_to_ind"][data[b]]][b] = 1

        return one_hot

    def create_chars(self, text):
        """ 
            Function to create a sequence of characters from a given one-hot-encoding. 
        """
        chars = ""
        for i in range(len(text)):
            pos = np.where(text[i] != 0)
            chars += self.data["ind_to_char"][pos[0][0]]

        return chars

    def adagrad(self):
        """ 
            Implementation of adagrad optimizer with parameter updates.
        """
        gradients = [self.grads["W"], self.grads["U"],
                     self.grads["V"], self.grads["b"], self.grads["c"]]
        m_values = [self.m_values["W"], self.m_values["U"],
                    self.m_values["V"], self.m_values["b"], self.m_values["c"]]
        params = [self.W, self.U, self.V, self.b, self.c]

        for i in range(len(params)):
            m_values[i] += (gradients[i] ** 2)

            params[i] += - ((self.eta / np.sqrt(m_values[i] +
                                                np.finfo(np.float32).eps)))*gradients[i]

    def rms_prop(self):
        """ 
            Implementation of RMSProp optimizer with parameter updates.
        """
        gradients = [self.grads["W"], self.grads["U"],
                     self.grads["V"], self.grads["b"], self.grads["c"]]
        m_values = [self.m_values["W"], self.m_values["U"],
                    self.m_values["V"], self.m_values["b"], self.m_values["c"]]
        params = [self.W, self.U, self.V, self.b, self.c]
        gamma = 0.9

        for i in range(len(params)):
            m_values[i] = gamma * m_values[i] + \
                (1 - gamma) * (gradients[i] ** 2)

            params[i] += - ((self.eta / np.sqrt(m_values[i] +
                                                np.finfo(np.float32).eps)))*gradients[i]

    def compare_gradients(self):
        """ 
            Main function for gradient comparison check.
        """
        position = 0
        x = self.data["book_data"][position:position+self.seq_length]
        y = self.data['book_data'][position +
                                   1:position+self.seq_length+1]

        x_one_hot = self.create_one_hot(x)
        y_one_hot = self.create_one_hot(y)
        h = np.zeros((self.m, 1))

        W1, V1, U1, b1, c1 = self.compute_gradients_analyt(
            x_one_hot, y_one_hot, h)
        self.reset_grads()
        W2, V2, U2, b2, c2 = self.compute_gradients_num(
            x_one_hot, y_one_hot, h)
        eps = 10**-6

        # mean absolute error
        print(np.mean(np.abs(b1 - b2)))
        print(np.mean(np.abs(c1 - c2)))
        print(np.mean(np.abs(W1 - W2)))
        print(np.mean(np.abs(U1 - U2)))
        print(np.mean(np.abs(V1 - V2)))

        # maximum relative error
        print(np.mean(np.abs(b1 - b2))/max(eps, np.mean(np.abs(b1) + np.abs(b2))))
        print(np.mean(np.abs(c1 - c2))/max(eps, np.mean(np.abs(c1) + np.abs(c2))))
        print(np.mean(np.abs(W1 - W2))/max(eps, np.mean(np.abs(W1) + np.abs(W2))))
        print(np.mean(np.abs(U1 - U2))/max(eps, np.mean(np.abs(U1) + np.abs(U2))))
        print(np.mean(np.abs(V1 - V2))/max(eps, np.mean(np.abs(V1) + np.abs(V2))))

    def plot_loss(self, iters, losses):
        """ 
            Plots loss across a given number of iterations.
        """
        iters = np.arange(iters)
        plt.figure(1)
        plt.plot(iters, losses, color='m')
        plt.xlabel('Iterations')
        plt.ylabel('Smooth Loss')
        plt.show()

    def train(self, epochs):
        """ 
            Main function for training process. Initializes necessary variables, and performs training across a given number of epochs. 
        """

        iters = 0
        maximum_iter = 1000
        losses = []
        smooth_loss = 0
        all_params = []

        for epoch in range(epochs):
            position = 0
            hprev = np.zeros((self.m, 1))

            while position < (len(self.data["book_data"]) - self.seq_length - 1):

                x = self.data["book_data"][position:position +
                                           self.seq_length]
                y = self.data['book_data'][position +
                                           1:position+self.seq_length+1]

                x_one_hot = self.create_one_hot(x)
                y_one_hot = self.create_one_hot(y)

                loss, hprev = self.compute_gradients(
                    x_one_hot, y_one_hot, hprev)

                if smooth_loss == 0:
                    smooth_loss = loss

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                losses.append(smooth_loss)
                all_params.append([self.grads["W"], self.grads["U"],
                                   self.grads["V"], self.grads["b"], self.grads["c"]])

                if iters == 0 or iters % 1000 == 0:
                    print("Iteration " + str(iters) + " " + "(epoch " +
                          str(epoch) + ")" + ": " + str(smooth_loss))

                    if iters == 0 or iters % 10000 == 0:
                        synth = self.synthesize_chars(
                            hprev, x_one_hot[:, 0], y_one_hot, 200)
                        generated_text = self.create_chars(synth)
                        print(generated_text)

                # Adagrad SGD
                self.adagrad()

                position += self.seq_length
                iters += 1

        lowest_loss = np.argmin(losses)
        best_params = all_params[lowest_loss]
        synth = self.synthesize_chars(
            hprev, x_one_hot[:, 0], y_one_hot, 1000)
        generated_text = self.create_chars(synth)
        print(generated_text)

        self.plot_loss(iters, losses)


def main(check=False):
    if not check:
        data = read_data()
        rnn = RNN(data, 100, 0.1, 25)
        rnn.train(3)
    else:
        data = read_data()
        rnn = RNN(data, 5, 0.1, 25)
        rnn.compare_gradients()


if __name__ == "__main__":
    main()
