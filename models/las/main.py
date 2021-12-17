import torch
from utils import * 

def main():
   h = torch.randn([32, 20, 128])
   h = h.to('cuda:0')

   layer = AttentionLSTMLayer(128, 1, embedding_dim=128, vocabulary_size=0) 
   layer = layer.to('cuda:0')
   s = torch.randn([32, 25, 128])
   s = s.to('cuda:0')

   h_out =  layer(s, h)
   print(h_out.shape)


if __name__ == '__main__':
    main()
