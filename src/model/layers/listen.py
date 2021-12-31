import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def forward(self, x, input_lengths):
        # input lengths should be on the cpu
        input_lengths = input_lengths.cpu()

        x = pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        blstm_data, _ = self.l_blstm(x)
        output, out_len = pad_packed_sequence(blstm_data, batch_first=True)
        return output, out_len

class StackedBLSTMLayer(torch.nn.Module):
    """
        Pyramidal stack of BLSTM, reference: https://arxiv.org/abs/1508.01211
        Is a stack of BLSTM layers, where in each successive layer the time 
        resolution is reduced by a factor of 2. 

        Total time resolution reduction: 2**num_layers

        --- 

        Args:
        
            input_size: Number of features of the input

            hidden_size: Number of features of the output and hidden layers

            num_layers: The number of BLSTM layer on top of the first one
        
        ---

        Shapes:

            input: Tensor of shape (N, L, H_in), batch first

            output: Tensor of shape (N, U, H_out)


            where: 
                * N: batch size

                * L: length of the sequence

                * H_in: input_size

                * H_out: hidden_size

                * U: L/(2**num_layers)
                
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackedBLSTMLayer, self).__init__()

        self.layers = torch.nn.ModuleList()

        self.layers += [BLSTMLayer(input_size, hidden_size)]
        self.layers += [BLSTMLayer(hidden_size*2, hidden_size) for i in range(0, num_layers)]

    def forward(self, sample, input_lengths):
        bs = sample.shape[0]
        next_out = sample
        next_len = input_lengths

        # padding of the input
        if next_out.shape[1] % 2:
            next_out = torch.nn.functional.pad(next_out, (0, 0, 0, 1), mode='reflect')
            next_len += 1

        for l in self.layers:
            output, out_len = l(next_out, next_len)
            # padding the input of the next layer
            if output.shape[1] % 2:
                output = torch.nn.functional.pad(output, (0, 0, 0, 1), mode='reflect')
                out_len += 1

            # aggreate h elements to reduce time resolution
            next_out = torch.reshape(output, (bs, -1, output.shape[-1]*2))
            next_len = torch.div(out_len, 2)

        return output
