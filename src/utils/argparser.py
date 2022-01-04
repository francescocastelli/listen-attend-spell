import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str,     
                        help="name of the model", required=True)
    parser.add_argument("--seed", help="seed for RNG", type=int, default=0)

    # parameters
    parser.add_argument("--bs", type=int, help="batch size", required=True)
    parser.add_argument("--lr", type=float, help="learning rate", required=True)
    parser.add_argument("--l2", type=float, help="weight decay", required=True)
    parser.add_argument("--lr_decay", type=float, help="gamma for exponential lr decay", required=True)
    parser.add_argument("--epochs", type=int, help="total number of epochs", required=True)

    # model
    parser.add_argument("--Mpath", type=str, 
                        help="path to the model to load (its relative to the cwd)", default=None)

    # logs args
    parser.add_argument("--console_logs", help="wheter to print stats during training", 
                        action="store_true")
    parser.add_argument("--logs", help="wheter to save logs and model", action="store_true")
    parser.add_argument("--ckpt", help="use the checkpoint stored in param.ckpt_path", action="store_true")
    parser.add_argument("--ckpt_rate", help="number of epochs on which ckpt are stored", type=int, default=0) 
    parser.add_argument("--embeddings_num", help="num of embeddings to store", type=int, default=0)

    # train args 
    parser.add_argument("--distributed", help="train on multiple gpus", action="store_true")
    parser.add_argument("--multitrain", help="if you want to train on multiple config", action="store_true")

    parser.add_argument("-w", type=int,     
                        help="number of workes for data loading", default=1)
    parser.add_argument("-d", type=str, 
                        help="device name", default=None)

    # front-end args
    parser.add_argument("--win_len_s", type=float, required=True)
    parser.add_argument("--hop_len_s", type=float, required=True)
    parser.add_argument("--sr", type=int, required=True)
    parser.add_argument("--f_min", type=float, required=True)
    parser.add_argument("--f_max", type=float, required=True)
    parser.add_argument("--pad", type=float, required=True)
    parser.add_argument("--n_mels", type=int, required=True)
    parser.add_argument("--power", type=int, required=True)
    
    return parser.parse_args()
