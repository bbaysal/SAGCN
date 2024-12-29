import torch
import numpy as np
from BiLSTM import MultiLayerBiLSTM
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd


class TwitterRawDataset:
    def __init__(self, file_path):
        """
        Load and parse the .raw Twitter dataset
        Args:
            file_path: Path to the .raw file
        """
        self.texts = []          # Original tweets
        self.targets = []        # Target entities
        self.labels = []         # Sentiment labels

        # Read and parse the raw file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process three lines at a time
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break

            tweet = lines[i].strip()
            target = lines[i + 1].strip()
            label = int(lines[i + 2].strip())

            # Convert -1, 0, 1 to 0, 1, 2 for classification
            label = label + 1 if label == -1 else label

            self.texts.append(tweet)
            self.targets.append(target)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'target': self.targets[idx],
            'label': self.labels[idx]
        }


class TwitterBertDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_length=128):
        """
        Prepare the dataset for BERT processing
        Args:
            raw_dataset: Instance of TwitterRawDataset
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]

        # Replace $T$ with the actual target
        text = item['text'].replace('$T$', item['target'])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


class BERTBiLSTMTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def get_bert_embeddings(self, batch) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            embeddings = outputs.last_hidden_state
            return embeddings

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        training_history = []
        device = torch.device("cude" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(device)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            batch_count = 0

            for batch in train_loader:
                # move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # get bert embeddings
                embeddings = self.get_bert_embeddings(batch)

                embeddings_np = embeddings.cpu().numpy()
                labels_np = batch["label"].cpu89.numpy()

                predictions = self.model.forward(embeddings_np)

                gradients, batch_loss = self.backward(
                    embeddings_np, predictions, labels_np)

                self.optimize(gradients)

                epoch_loss += batch_loss
                batch_count += 1

            val_loss = self.evaluate(val_loader, device)

            avg_epoch_loss = epoch_loss / batch_count

            training_history.append({
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss
            })

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss: .4f}, Validation Loss: {val_loss: .4f}")

            return training_history

    def evaluate(self, val_loader: DataLoader, device: torch.device) -> float:
        self.model.eval()
        total_loss = 0
        batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                embeddings = self.get_bert_embeddings(batch)
                embeddings_np = embeddings.cpu().numpy()
                labels_np = batch["labels"].cpu().numpy()

                predictions = self.model.forward(embeddings_np)
                loss = self.calculate_loss(predictions, labels_np)

                total_loss += loss
                batch_count += 1

        return total_loss / batch_count


def prepare_twitter_data(file_path, batch_size=32, max_length=128):
    """
    Prepare the Twitter data for training
    Args:
        file_path: Path to the .raw file
        batch_size: Size of training batches
        max_length: Maximum sequence length
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load raw dataset
    raw_dataset = TwitterRawDataset(file_path)

    # Create BERT dataset
    bert_dataset = TwitterBertDataset(raw_dataset, tokenizer, max_length)

    # Split into train and validation sets
    train_size = int(0.8 * len(bert_dataset))
    val_size = len(bert_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        bert_dataset,
        [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    input_dim = 768
    hidden_dim = 300
    output_dim = 3

    model = MultiLayerBiLSTM(input_dim, hidden_dim, output_dim, 2)
    trainer = BERTBiLSTMTrainer(model)

    train_loader, val_loader = prepare_twitter_data("dataset/train.raw")

    history = trainer.train(train_loader, val_loader, epochs=10)
