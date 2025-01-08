#!/bin/bash


# lr=('0.01' '0.003' '0.001' '0.0001')
# reg_weight=('0.01' '0.001' '0.0003' '0.0001')
emb_size=(64)
# lr=('1e-2' '3e-3' '1e-3' '1e-4')
lr=('1e-4')
reg_weight=('1e-2' '1e-3' '3e-4' '1e-4')

#reg_weight=('1e-4')
gpu='cuda:4'

dataset=('UB_2')
# shellcheck disable=SC2068
for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
                echo 'start train: '$name 'lr: '$l 'reg: '$reg 'emb: '$emb gpu: $gpu
                `
                    nohup python main.py \
                        --lr ${l} \
                        --reg_weight ${reg} \
                        --data_name $name \
                        --device $gpu \
                        --stage 2 \
                        --embedding_size $emb > "./nohup_log/${name}/lr_${l}_reg_${reg}_emb_${emb}.log" 2>&1 &
                `
                echo 'train end: '$name 'lr: '$l 'reg: '$reg 'emb: '$emb gpu: $gpu
            done
        done
    done
done