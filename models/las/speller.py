import torch

class Speller(torch.nn.Module):
    
    def __init__(self, hidden_size, num_layers):
        super(Speller, self).__init__()

        self.cells = torch.nn.ModuleList(
                        [torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

    def zero_rnn(self, shape):
       return torch.zeros(shape, device='cuda:0'), torch.zeros(shape, device='cuda:0') 

    def forward(self, x):
        bs = x.shape[0]
        seq_len = x.shape[1]

        h_t = [x[:, 0, :]]
        c_t = [zero_rnn((bs, self.hidden_size))[1]]
        for i in range(len(self.cells)):
            h, c = zero_rnn((bs, self.hidden_size))
            h_t.append(h)
            c_t.append(c)

        for t in range(seq_len):
            h_t[0] = x[:, t, :]
            for i, l in enumerate(self.cells, 1):
                h_t[i], c_t[i] = l(h_t[i-1], (h_t[i], c_t[i]))

        rnn_out = h_t[-1] 
