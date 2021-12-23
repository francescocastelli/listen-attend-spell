#!/bin/bash
    
# mandatory parameters

name="las"
workers_num=4
bs=64
lr=0.0003
l2=0.5
lr_decay=0.5
epochs=100
seed=1000

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
# distributed=true
# multitrain=true
# embeddings=true

# train command

python3 model/train.py --name ${name} -w ${workers_num} --bs ${bs} --lr ${lr} \
		       --lr_decay ${lr_decay} --epochs ${epochs} --l2 ${l2} --seed ${seed} \
		       --win_len_s ${win_len_s} --hop_len_s ${hop_len_s} --sr ${sr} \
		       --f_min ${f_min} --f_max ${f_max} --pad ${pad} --n_mels ${n_mels} --power ${power} \
		       ${console_logs:+--console_logs} ${logs:+--logs} \
		       ${distributed:+--distributed} ${embeddings:+--embeddings} ${multitrain:+--multitrain}

