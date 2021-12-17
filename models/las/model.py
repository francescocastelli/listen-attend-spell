import torch
import torch.nn as nn
from torchtrainer.model import Model 
from utils import StackedBLSTMLayer

class LAS(Model):
    def __init__(self, name):
        super().__init__(name=name)
        
        self.encoder = StackedBLSTMLayer(h_in, h_out, 3)

    def define_optimizer_scheduler(self):
        pass

    def embeddings_forward(self, sample):
        pass

    def forward(self, sample):
        pass

    def training_step(self, sample):
        pass

    def validation_step(self, sample):
        pass
