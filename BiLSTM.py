import numpy as np


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier initialization for weights and zero initialization for biases
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))

        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))

        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.vstack((h_prev, x))

        # Forget gate
        ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)

        # Input gate
        it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)

        # Candidate cell state
        c_tilde = self.tanh(np.dot(self.Wc, combined) + self.bc)

        # Current cell state
        ct = ft * c_prev + it * c_tilde

        # Output gate
        ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)

        # Current hidden state
        ht = ot * self.tanh(ct)

        return ht, ct


class BiLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim

        # Initialize forward and backward LSTM cells
        self.forward_lstm = LSTMCell(input_dim, hidden_dim // 2)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim // 2)

        # Output layer weights
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros((output_dim, 1))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def forward(self, X):
        batch_size, seq_len, input_dim = X.shape
        outputs = []

        for b in range(batch_size):
            h_forward = np.zeros((self.hidden_dim // 2, 1))
            c_forward = np.zeros((self.hidden_dim // 2, 1))

            h_backward = np.zeros((self.hidden_dim // 2, 1))
            c_backward = np.zeros((self.hidden_dim // 2, 1))

            forward_outputs = []
            backward_outputs = []

            # Forward pass
            for t in range(seq_len):
                x = X[b, t, :].reshape(-1, 1)
                h_forward, c_forward = self.forward_lstm.forward(
                    x, h_forward, c_forward)
                forward_outputs.append(h_forward)

            # Backward pass
            for t in reversed(range(seq_len)):
                x = X[b, t, :].reshape(-1, 1)
                h_backward, c_backward = self.backward_lstm.forward(
                    x, h_backward, c_backward)
                backward_outputs.insert(0, h_backward)

            # Concatenate forward and backward outputs
            for f_out, b_out in zip(forward_outputs, backward_outputs):
                combined = np.vstack((f_out, b_out))
                y = np.dot(self.Wy, combined) + self.by
                outputs.append(self.softmax(y))

        return outputs


class MultiLayerBiLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.num_layers = num_layers
        self.bilstm_layers = []

        # Initialize BiLSTM layers
        for i in range(num_layers):
            if i == 0:
                self.bilstm_layers.append(
                    BiLSTM(input_dim, hidden_dim, hidden_dim))
            else:
                self.bilstm_layers.append(
                    BiLSTM(hidden_dim, hidden_dim, hidden_dim))

        # Output layer weights
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros((output_dim, 1))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def forward(self, X):
        batch_size, seq_len, input_dim = X.shape
        current_input = X

        # Process through each BiLSTM layer
        for layer in self.layers:
            current_input = layer.forward(current_input)

        # Final output layer
        outputs = []
        for b in range(batch_size):
            sequence_outputs = []
            for t in range(seq_len):
                hidden = current_input[b, t].reshape(-1, 1)
                y = np.dot(self.Wy, hidden) + self.by
                sequence_outputs.append(self.softmax(y))
            outputs.append(sequence_outputs)

        return outputs
