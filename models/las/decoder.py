import torch
from utils import AttentionLSTMLayer

class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, vocabulary_size, embedding_dim):
        super(Decoder, self).__init__()

        self.att_rnn = AttentionLSTMLayer(hidden_size, embedding_dim)
        self.rnns = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True,
                                  num_layers=num_layers)
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)


    def forward(self, y, encoder_h):
        y_emb = self.embeddings(y)
        h_att, c_att = self.att_rnn(y_emb, encoder_h)
        h_out, _ = self.






