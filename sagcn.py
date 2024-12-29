import torch
import torch.nn as nn
from transformers import BertModel
from BiLSTM import BiLSTM


class SAGCN(nn.Module):
    def __init__(self, bert_path, hidden_dim=300, num_layers=2, num_heads=8, dropout=0.3, num_classes=3):
        super(SAGCN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_dim = 768
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bi_lstm = BiLSTM(self.bert_dim, hidden_dim,
                              hidden_dim, num_layers)
