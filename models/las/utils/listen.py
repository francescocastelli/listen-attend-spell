import torch

class BLSTMLayer(torch.nn.Module):
    """ 
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            raise ValueError(f"output_dim of BLSTMLayer is {output_dim}, "
                             f"but expect a even layer size")

        # bi-directional LSTM
        self.l_blstm = torch.nn.LSTM(input_dim, output_dim // 2, 
                                     bidirectional=True, batch_first=True)

    def forward(self, x):
        blstm_data, _ = self.l_blstm(x)
        return blstm_data

class StackedBLSTMLayer(torch.nn.Module):
    """
        Args:
        
        input_size: H_in
        hidden_size: H_out
        num_layers: The number of BLSTM layer on top of the first one
        
        ---------

        Input: Tensor of shape (N, L, H_in), batch first
        Ouput: Tensor of shape (N, L/(2**num_layers), H_out)
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackedBLSTMLayer, self).__init__()

        self.layers = torch.nn.ModuleList()

        self.layers += [BLSTMLayer(input_size, hidden_size)]
        self.layers += [BLSTMLayer(hidden_size*2, hidden_size) for i in range(0, num_layers)]

    def forward(self, sample):
        bs = sample.shape[0]

        next_out = sample
        for l in self.layers:
            output = l(next_out)
            # aggreate h elements to reduce time resolution
            next_out = torch.reshape(output, (bs, -1, output.shape[-1]*2))

        return output
