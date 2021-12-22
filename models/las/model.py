import torch
import torch.nn as nn
from torchtrainer.model import Model 
from layers.listen import StackedBLSTMLayer
from layers.attendspell import AttendAndSpell 
from tokenizer import pad_value

class LAS(Model):
    def __init__(self, name, input_size, hidden_size, embedding_dim, vocabulary_size, args):

        super().__init__(name=name)

        self.lr = args.lr
        self.l2 = args.l2
        self.lr_decay = args.lr_decay
        
        self.encoder = StackedBLSTMLayer(input_size, hidden_size, 3)
        self.decoder = AttendAndSpell(hidden_size, embedding_dim, vocabulary_size, 1)

        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_value)

    def define_optimizer_scheduler(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, gamma=self.lr_decay, 
                                                    step_size=5, last_epoch=-1)
        return opt, scheduler 

    def embeddings_forward(self, sample):
        pass

    def forward(self, sample):
        melspec, input_lengths, y_emb = sample

        encoder_h = self.encoder(melspec, input_lengths)

        y_out = self.decoder(y_emb, encoder_h)
        return y_out

    def training_step(self, sample):
        melspec = sample['melspec']
        input_lengths = sample['lengths']
        y = sample['token_seq']

        # compute the embeddings for y
        y_emb = self.embeddings(y)
        
        # get output seq 
        y_out = self((melspec, input_lengths, y_emb)) 
        
        # compute loss function
        loss = self.loss(y_out, y.view(-1))

        # metrics
        with torch.no_grad():
            running_loss = loss * melspec.shape[0]
            self.save_train_stats(loss_train=running_loss) 

        return loss


    def validation_step(self, sample):
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
        self.save_train_stats(loss_train=running_loss) 

        return loss
