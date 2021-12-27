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
    def __init__(self, hidden_size, embedding_dim, vocabulary_size, num_layers):
        super(AttendAndSpell, self).__init__()
        
        self.vocab_size = vocabulary_size
        self.att_rnn = torch.nn.LSTMCell(hidden_size + embedding_dim, hidden_size)
        self.rnns = torch.nn.ModuleList(
                      [torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)]
                     )


        self.attention = AttentionContext(hidden_size)
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_size*2, self.hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden_size, self.vocab_size))

        # look-up table for char embeddings
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)

        self.d = torch.nn.Parameter(torch.empty(0))

    def zero_rnn(self, shape):
       return torch.zeros(shape, device=self.d.device), torch.zeros(shape, device=self.d.device) 

    def init_out(self, shape):
        h_t = [self.zero_rnn(shape)[0]]
        c_t = [self.zero_rnn(shape)[1]]
        for i in range(len(self.rnns)):
            h, c = self.zero_rnn(shape)
            h_t.append(h)
            c_t.append(c)

        return h_t, c_t

    def forward(self, y, encoder_h):
        bs = y.shape[0]
        seq_len = y.shape[1]

        h_t, c_t = self.init_out((bs, self.hidden_size))
        att_i = h_t[0]
        
        # compute the embeddings for y
        y_emb = self.embeddings(y)

        y_out = []
        # over the seq len
        for i in range(seq_len):
            rnn_in = torch.cat((y_emb[:, i, :], att_i), dim=-1)
            h_t[0], c_t[0]  = self.att_rnn(rnn_in, (h_t[0], c_t[0]))

            for i, l in enumerate(self.rnns, 1):
                h_t[i], c_t[i] = l(h_t[i-1], (h_t[i], c_t[i]))

            att_i = self.attention(h_t[-1], encoder_h)
            mlp_in = torch.cat((h_t[-1], att_i), dim=-1)
            y_pred_i = self.mlp(mlp_in)
            y_out.append(torch.unsqueeze(y_pred_i, dim=1))

        y_out = torch.hstack(y_out) 
        y_out = torch.reshape(y_out, (bs*seq_len, self.vocab_size))
        return y_out

    def inference(self, encoder_h, sos_tok, beam_width=2, max_len=40):
        # zero the first lstm state
        h_t, c_t = self.init_out((bs, self.hidden_size))
        att_i = h_t[0]

        hyp_0 = {'seq': [sos_tok], 'score': 0.0, 'h': h_t, 'c': c_t}
        hypothesis = [hyp_0]
        for i in range(max_len):

            hyps_best = []
            for hyp_i in hypothesis:
                # take the last predicted char in the current hypotesis
                y_i = hyp_i['seq'][i]
                # take the hidden state and context of the current hypotesis
                h_t, c_t = hyp_i['h'], hyp_i['c']

                # compute embedding and feed it to the decoder
                y_emb = self.embeddings(y_i)
                
                # single time step of the decoder
                rnn_in = torch.cat((y_emb, att_i), dim=-1)
                h_t[0], c_t[0]  = self.att_rnn(rnn_in, (h_t[0], c_t[0]))

                for i, l in enumerate(self.rnns, 1):
                    h_t[i], c_t[i] = l(h_t[i-1], (h_t[i], c_t[i]))

                att_i = self.attention(h_t[-1], encoder_h)
                mlp_in = torch.cat((h_t[-1], att_i), dim=-1)
                y_pred_i = self.mlp(mlp_in)
                y_out.append(torch.unsqueeze(y_pred_i, dim=1))

                # compute the score of each char in the vocab
                y_scores = torch.nn.functional.log_softmax(y_out, dim=-1)
                y_top_scores, y_top_indices = torch.topk(y_scores, beam_width)

                # add the top scores at the current hypothesis
                for b in range(beam_width):
                    char, score = y_scores[y_top_indices[b]], y_top_scores[b]
                    seq = hyp_i['seq'].append(char)
                    score = hyp_i['score'] + score
                    hyp = {'h': h_t, 'c': c_t, 'seq': seq, 'score': score}
                    hyps_best.append(hyp)
          
                hyps_best = sorted(hyps_best, key=lambda d: d['score'], reverse=True)[:beam_width] 
            
            # keep only the best hypothesis
            hypothesis = hyps_best

        # end of beam search
        return hypothesis
