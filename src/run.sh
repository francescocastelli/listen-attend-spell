#!/bin/bash
    
# mandatory parameters

name="las"
workers_num=4
bs=32
lr=0.02
l2=0.0
lr_decay=0.98
epochs=150
seed=1000
ckpt_rate=0
embeddings_num=0

# front end parameters
win_len_s=0.025
hop_len_s=0.010
sr=16000
f_min=20
f_max=7500
pad=0
n_mels=64
power=2

# optional parameters, comments a line if you dont want it

console_logs=true
#logs=true
#ckpt=true
# distributed=true
# multitrain=true

# train command

python3 train.py --name ${name} -w ${workers_num} --bs ${bs} --lr ${lr} \
		 --lr_decay ${lr_decay} --epochs ${epochs} --l2 ${l2} --seed ${seed} \
		 --win_len_s ${win_len_s} --hop_len_s ${hop_len_s} --sr ${sr} \
		 --f_min ${f_min} --f_max ${f_max} --pad ${pad} --n_mels ${n_mels} --power ${power} \
		 --embeddings_num ${embeddings_num} \
		 ${console_logs:+--console_logs} ${logs:+--logs} ${ckpt:+--ckpt} --ckpt_rate ${ckpt_rate}\
		 ${distributed:+--distributed} ${embeddings:+--embeddings} ${multitrain:+--multitrain}

