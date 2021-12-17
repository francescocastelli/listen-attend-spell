import torch

class AttentionContext(torch.nn.Module):
    """
        Args: 

        hidden_size: H_out used in the encoder and decoder 
        
        ---------

        inputs: 
            s: The decoder current state, shape: (bs, hidden_size)
            h: The entire output state seq of the encoder, shape: (bs, L, hidden_size)

        output: Context vector for every element in the batch, shape: (bs, hidden_size) 
    """
    def __init__(self, hidden_size): 
        super(AttentionContext, self).__init__()
        self.mlp_s = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp_h = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # along dim -1 -> along the time seq of the encoder
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, s, h):
        s_e, h_e = self.mlp_s(s), self.mlp_h(h)

        # after unsqueeze and transpose (before batch mm) I have this:
        # s_e is the decoder state at a specific time in the seq (bs, 1, hidden_size)
        # h_e is the entire encoder state (bs, hidden_size, L)
        e = torch.bmm(torch.unsqueeze(s_e, 1), torch.transpose(h_e, 1, -1))
        
        # a is (bs, 1, L), dim = -1 sums up to 1
        a = self.softmax(e)
        
        # c is (bs, 1, hidden_size), 
        # it contains the rescaled (using a, over the encoder seq len) encoder state
        c = torch.bmm(a, h_e) 
        return torch.squeeze(c, 1)

class AttendAndSpell(torch.nn.Module):
    """
        Args:
        
        hidden_size: H_out
        embedding_dim: Embedding size of the char in the input sequence
        
        ---------

        inputs: 
            y: The entire input seq of the decoder, shape: (bs, S, hidden_size)
               This layer expects embeddings for each char

            encoder_h: The entire output state seq of the encoder, shape: (bs, L, hidden_size)

        Ouput: The sequence of hidden states from the output of the attention lstm, shape: (bs, S, hidden_size)
    """
    def __init__(self, hidden_size, embedding_dim, vocabulary_size):
        super(AttendAndSpell, self).__init__()
        
        self.att_rnn = torch.nn.LSTMCell(hidden_size + embedding_dim, hidden_size)
        self.rnns = torch.nn.ModuleList(
                      [torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)]
                     )


        self.attention = AttentionContext(hidden_size)
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_size*2, self.hidden_size),
                        torch.nn.ReLu(),
                        torch.nn.Linear(self.hidden_size, vocabulary_size))

    def zero_rnn(self, shape):
       return torch.zeros(shape, device='cuda:0'), torch.zeros(shape, device='cuda:0') 

    def init_out(self):
        h_t = [self.zero_rnn(bs, self.hidden_size))[0]]
        c_t = [self.zero_rnn((bs, self.hidden_size))[1]]
        for i in range(len(self.cells)):
            h, c = self.zero_rnn((bs, self.hidden_size))
            h_t.append(h)
            c_t.append(c)

    def forward(self, y, encoder_h):
        seq_len = y.shape[1]
        h_i, c_i = self.zero_rnn((y.shape[0], self.hidden_size))
        att_i = h_i

        h_t, c_t = self.init_out()
        
        y_out = []
        # over the seq len
        for i in range(seq_len):
            rnn_in = torch.cat((y[:, i, :], att_i), dim=-1)
            h_t[0], c_i  = self.att_rnn(rnn_in, (h_i, c_i))

            for i, l in enumerate(self.cells, 1):
                h_t[i], c_t[i] = l(h_t[i-1], (h_t[i], c_t[i]))

            att_i = self.attention(h_t[-1], encoder_h)
            mpl_in = torch.cat((h_t[-1], att_i))
            y_pred_i = self.mlp(mlp_in)
            y_out.append(torch.unsqueeze(y_pred_i, dim=1))

        return torch.hstack(y_out)
