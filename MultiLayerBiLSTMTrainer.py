import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Dict
import time
import matplotlib.pyplot as plt
from BiLSTM import MultiLayerBiLSTM
from tqdm import tqdm


class TwitterMultiLayerBiLSTMSystem:
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, learning_rate: float = 0.01):
        """
        Initialize the complete system for Twitter sentiment analysis
        Args:
            hidden_dim: Dimension of hidden layers
            num_layers: Number of BiLSTM layers
            learning_rate: Learning rate for optimization
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Initialize BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # Initialize BiLSTM model
        self.input_dim = 768  # BERT embedding dimension
        self.output_dim = 3   # Three sentiment classes
        self.model = MultiLayerBiLSTM(
            self.input_dim, hidden_dim, self.output_dim, num_layers)

    def load_and_process_data(self, file_path: str, max_length: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process the raw Twitter dataset
        Args:
            file_path: Path to the .raw file
            max_length: Maximum sequence length for tokenization
        Returns:
            Tuple of (embeddings, labels)
        """
        # Read raw file
        texts, targets, labels = [], [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break

            texts.append(lines[i].strip())
            targets.append(lines[i + 1].strip())
            labels.append(int(lines[i + 2].strip()))

        # Process data
        embeddings = []
        processed_labels = []
        label_map = {-1: 0, 0: 1, 1: 2}  # Convert labels to 0,1,2

        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(device)
        index = 0
        for text, target, label in tqdm(zip(texts, targets, labels), 
                                        desc="Data Preparing", 
                                        total=len(texts)):
            # Replace $T$ with actual target
            full_text = text.replace('$T$', target)

            # Tokenize
            encoded = self.tokenizer(
                full_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Get BERT embeddings
            with torch.no_grad():
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = self.bert_model(**encoded)
                embedding = outputs.last_hidden_state.cpu().numpy()
                # Take first (and only) sequence
                embeddings.append(embedding[0])

            # Process label
            processed_labels.append(label_map[label])

        return np.array(embeddings), np.eye(3)[processed_labels]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32,
              validation_split: float = 0.2):
        """
        Train the MultiLayerBiLSTM model
        Args:
            X: Input data (BERT embeddings)
            y: Labels
            epochs: Number of training epochs
            batch_size: Size of training batches
            validation_split: Fraction of data to use for validation
        """
        # Split data into train and validation sets
        train_size = int((1 - validation_split) * len(X))
        indices = np.random.permutation(len(X))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Initialize trainer
        trainer = MultiLayerBiLSTMTrainer(self.model, self.learning_rate)

        # Train model
        history = trainer.train(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate model performance
        Args:
            X: Test data
            y: True labels
        """
        predictions = self.model.forward(X)
        predictions = np.array([pred.argmax() for pred in predictions])
        true_labels = y.argmax(axis=1)

        accuracy = np.mean(predictions == true_labels)

        # Calculate metrics per class
        classes = ['Negative', 'Neutral', 'Positive']
        class_metrics = {}

        for i, class_name in enumerate(classes):
            class_mask = true_labels == i
            class_accuracy = np.mean(
                predictions[class_mask] == true_labels[class_mask])
            class_metrics[class_name] = class_accuracy
            print(f"{class_name} Accuracy: {class_accuracy:.4f}")

        print(f"Overall Accuracy: {accuracy:.4f}")
        return accuracy, class_metrics

    def plot_training_history(self, history: Dict):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.show()

    def predict(self, text: str, target: str) -> str:
        """
        Make prediction for a single text
        Args:
            text: Input text with $T$ placeholder
            target: Target entity
        Returns:
            Predicted sentiment (Negative/Neutral/Positive)
        """
        # Replace placeholder and get embeddings
        full_text = text.replace('$T$', target)

        # Tokenize and get BERT embeddings
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get BERT embeddings
        with torch.no_grad():
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = self.bert_model(**encoded)
            embedding = outputs.last_hidden_state.cpu().numpy()

        # Get prediction
        prediction = self.model.forward(embedding[None, ...])[0]
        pred_class = prediction.argmax()

        # Map prediction to sentiment
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[pred_class]


class MultiLayerBiLSTMTrainer:
    def __init__(self, model: MultiLayerBiLSTM, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate

    def calculate_loss(self, predictions: List[np.ndarray], targets: np.ndarray) -> float:
        """
        Calculate cross-entropy loss
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        total_loss = 0
        n_samples = len(predictions)

        for pred, target in zip(predictions, targets):
            pred = np.clip(pred, epsilon, 1 - epsilon)
            total_loss += -np.sum(target * np.log(pred))

        return total_loss / n_samples

    def backward(self, X: np.ndarray, predictions: List[np.ndarray],
                 targets: np.ndarray) -> Tuple[Dict, float]:
        """
        Compute gradients using backpropagation through time (BPTT)
        """
        gradients = {}
        # Initialize gradients for each layer
        for i in range(self.model.num_layers):
            layer_prefix = f'layer_{i}_'
            gradients.update({
                f'{layer_prefix}Wf_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.Wf),
                f'{layer_prefix}Wi_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.Wi),
                f'{layer_prefix}Wo_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.Wo),
                f'{layer_prefix}Wc_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.Wc),
                f'{layer_prefix}bf_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.bf),
                f'{layer_prefix}bi_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.bi),
                f'{layer_prefix}bo_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.bo),
                f'{layer_prefix}bc_forward': np.zeros_like(self.model.bilstm_layers[i].forward_lstm.bc),

                f'{layer_prefix}Wf_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.Wf),
                f'{layer_prefix}Wi_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.Wi),
                f'{layer_prefix}Wo_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.Wo),
                f'{layer_prefix}Wc_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.Wc),
                f'{layer_prefix}bf_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.bf),
                f'{layer_prefix}bi_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.bi),
                f'{layer_prefix}bo_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.bo),
                f'{layer_prefix}bc_backward': np.zeros_like(self.model.bilstm_layers[i].backward_lstm.bc),
            })

        # Output layer gradients
        gradients.update({
            'Wy': np.zeros_like(self.model.Wy),
            'by': np.zeros_like(self.model.by)
        })

        loss = self.calculate_loss(predictions, targets)

        # Calculate output layer gradients
        for pred, target in zip(predictions, targets):
            error = pred - target
            gradients['Wy'] += np.dot(error, pred.T)
            gradients['by'] += error

        # Normalize gradients by batch size
        for key in gradients:
            gradients[key] /= len(predictions)

        return gradients, loss

    def optimize(self, gradients: Dict):
        """
        Update model parameters using calculated gradients
        """
        # Update each layer's parameters
        for i in range(self.model.num_layers):
            layer = self.model.bilstm_layers[i]
            layer_prefix = f'layer_{i}_'

            # Update forward LSTM
            layer.forward_lstm.Wf -= self.learning_rate * \
                gradients[f'{layer_prefix}Wf_forward']
            layer.forward_lstm.Wi -= self.learning_rate * \
                gradients[f'{layer_prefix}Wi_forward']
            layer.forward_lstm.Wo -= self.learning_rate * \
                gradients[f'{layer_prefix}Wo_forward']
            layer.forward_lstm.Wc -= self.learning_rate * \
                gradients[f'{layer_prefix}Wc_forward']
            layer.forward_lstm.bf -= self.learning_rate * \
                gradients[f'{layer_prefix}bf_forward']
            layer.forward_lstm.bi -= self.learning_rate * \
                gradients[f'{layer_prefix}bi_forward']
            layer.forward_lstm.bo -= self.learning_rate * \
                gradients[f'{layer_prefix}bo_forward']
            layer.forward_lstm.bc -= self.learning_rate * \
                gradients[f'{layer_prefix}bc_forward']

            # Update backward LSTM
            layer.backward_lstm.Wf -= self.learning_rate * \
                gradients[f'{layer_prefix}Wf_backward']
            layer.backward_lstm.Wi -= self.learning_rate * \
                gradients[f'{layer_prefix}Wi_backward']
            layer.backward_lstm.Wo -= self.learning_rate * \
                gradients[f'{layer_prefix}Wo_backward']
            layer.backward_lstm.Wc -= self.learning_rate * \
                gradients[f'{layer_prefix}Wc_backward']
            layer.backward_lstm.bf -= self.learning_rate * \
                gradients[f'{layer_prefix}bf_backward']
            layer.backward_lstm.bi -= self.learning_rate * \
                gradients[f'{layer_prefix}bi_backward']
            layer.backward_lstm.bo -= self.learning_rate * \
                gradients[f'{layer_prefix}bo_backward']
            layer.backward_lstm.bc -= self.learning_rate * \
                gradients[f'{layer_prefix}bc_backward']

        # Update output layer
        self.model.Wy -= self.learning_rate * gradients['Wy']
        self.model.by -= self.learning_rate * gradients['by']

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
              validation_data: Tuple[np.ndarray, np.ndarray] = None):
        """
        Train the MultiLayerBiLSTM model
        Args:
            X: Training data of shape (n_samples, seq_length, input_dim)
            y: Target labels of shape (n_samples, num_classes)
            epochs: Number of training epochs
            batch_size: Size of training batches
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        n_samples = X.shape[0]
        training_history = {
            'train_loss': [],
            'val_loss': [] if validation_data is not None else None
        }

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0

            # Create mini-batches
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass
                predictions = self.model.forward(X_batch)

                # Backward pass
                gradients, batch_loss = self.backward(
                    X_batch, predictions, y_batch)

                # Update parameters
                self.optimize(gradients)

                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / (n_samples // batch_size)
            training_history['train_loss'].append(avg_epoch_loss)

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self.model.forward(X_val)
                val_loss = self.calculate_loss(val_predictions, y_val)
                training_history['val_loss'].append(val_loss)

                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

        return training_history


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = TwitterMultiLayerBiLSTMSystem(
        hidden_dim=256,
        num_layers=2,
        learning_rate=0.01
    )

    # Load and process data
    X, y = system.load_and_process_data('dataset/train.raw')

    # Train model
    history = system.train(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Evaluate model
    print("\nModel Evaluation:")
    accuracy, class_metrics = system.evaluate(X, y)

    # Plot training history
    system.plot_training_history(history)

    # Example prediction
    sample_text = "The new update for $T$ is terrible!"
    sample_target = "Windows 10"
    prediction = system.predict(sample_text, sample_target)
    print(f"\nSample Prediction:")
    print(f"Text: {sample_text.replace('$T$', sample_target)}")
    print(f"Predicted Sentiment: {prediction}")
