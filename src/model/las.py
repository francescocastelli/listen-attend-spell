import torch
from torchtrainer.model import Model 
from model.layers.listen import StackedBLSTMLayer
from model.layers.attendspell import AttendAndSpell 
from utils.tokenizer import pad_value

class LAS(Model):
    r"""
        Listen Attend and Spell model, reference: https://arxiv.org/abs/1508.01211
        Encoder-Decoder model, where: 
            
            * Encoder: The encoder is called Listener, and is an acoustic encoder model.
                       Its job is to map a sequence of acoustic feature in input to an
                       hidden representation with higher dimesionality and lower length.
                       Its implemented as a pyramidal stack of BLSTM layers.
                        
            * Decoder: The decoder is also called AttendAndSpell, and is an attention-based
                       character decoder. Its job is to map the output of the encoder to a
                       probability distribution over the characters sequences.
                       Its implemented as a multi-layer LSTM with attention.

        ---

        Args: 
            name: Name of the model

            input_size: Number of expected feature in the input

            hidden_size: Number of features in the hidden layers

            encoder_layers: Number of layers in the encoder pBLSTM

            decoder_layers: Number of layers in the decoder LSTM

            embedding_dim: Dimension of the char embeddings

            vocabulary_size: Number of possible characters

            args: All the other parameters of the model (lr, l2, lr_decay)


        Inputs:
            sample (tuple): tuple of (melspec, input_lengths, y_emb)

            where: 
                * melspec is the sequence of acoustic features

                * input_lengths containse the length of each melspec in the batch

                * y_emb is the sequence of embeddings of characters 
            
        Shape: 
            sample: 
                * melspec (bs, T, input_size)

                * input_lengths (bs) 
                    
                * y_emb (bs, S, embedding_dim)

            where: 
                * T is the length of the acoustic feature vector

                * S is the length of the sequence of char

    """
    def __init__(self, name, input_size, hidden_size, encoder_layers, decoder_layers,
                 embedding_dim, vocabulary_size, tokenizer, inference=False, args=None):
        super().__init__(name=name)
        
        if not inference and args is None:
            raise ValueError("You must specify some args for training")

        if not inference:
            self.lr = args.lr
            self.l2 = args.l2
            self.lr_decay = args.lr_decay
        
        self.tokenizer = tokenizer

        # seq2seq model
        self.encoder = StackedBLSTMLayer(input_size, hidden_size, encoder_layers)
        self.decoder = AttendAndSpell(hidden_size, embedding_dim, 
                                      vocabulary_size, decoder_layers)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_value)
        

    def define_optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_decay, 
                                                           last_epoch=-1)

        return opt, scheduler 

    def embeddings_forward(self, sample):
        pass

    def forward(self, sample):
        melspec, input_lengths, y_in = sample

        encoder_h = self.encoder(melspec, input_lengths)
        y_pred = self.decoder(y_in, encoder_h)
        return y_pred

    def training_step(self, sample):
        melspec = sample['melspec']
        input_lengths = sample['lengths']
        y_in, y_out = sample['y_in'], sample['y_out']

        # get output seq 
        y_pred = self((melspec, input_lengths, y_in)) 

        # compute loss function
        loss = self.loss(y_pred, y_out.view(-1))

        # metrics
        with torch.no_grad():
            #y_pred = torch.argmax(y_pred, dim=1)
            #cer = char_error_rate(y_pred, y.view(-1))
            running_loss = loss * melspec.shape[0]
            self.save_train_stats(loss_train=running_loss)#, char_error_rate_train=cer) 

        return loss

    def validation_step(self, sample):
        """
        melspec = sample['melspec']
        input_lengths = sample['lengths']
        y = sample['token_seq']

        # compute the embeddings for y
        y_emb = self.embeddings(y)
        
        # get output seq 
        y_out = self((melspec, input_lengths, y_emb)) 
        
        # compute loss function
        loss = self.loss(y_out, y)

        # metrics
        running_loss = loss * melspec.shape[0]
        self.save_train_stats(loss_valid=running_loss) 

        return loss
        """
        pass

    def inference_step(self, melspec, seq_len):
        self.eval()
        with torch.no_grad():
            # encoder step
            encoder_h = self.encoder(melspec, seq_len)

            # beam seach
            sos_tok = self.tokenizer.get_sos_token()
            hypothesis = self.decoder.inference(encoder_h, sos_tok)
            pred_seq = [hyp['seq'] for hyp in hypothesis]
            pred_decoded = [self.tokenizer.decode_tokens(s)[1] for s in pred_seq]

        return pred_decoded

