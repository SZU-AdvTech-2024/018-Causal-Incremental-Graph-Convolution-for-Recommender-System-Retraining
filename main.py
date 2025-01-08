#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: yanms
# @Date  : 2021/11/1 15:25
# @Desc  :
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from model import I_CRGCN
import os
from os.path import join

from data_set import DataSet
from trainer import Trainer

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)

    parser.add_argument('--data_name', type=str, default='jdata', help='')
    parser.add_argument('--stage', type=int, default=4, help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=True, help='')

    # parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--topk', type=list, default=[5, 10, 20, 50], help='')
    # parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--metrics', type=list, default=['ndcg', 'recall'], help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=3072, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=20000, help='')
    parser.add_argument('--model_path', type=str, default='./check_point/', help='')
    parser.add_argument('--emb_saved_path', type=str, default='./embeddings_save/', help='')
    parser.add_argument('--degree_path', type=str, default='/degree', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cpu', help='')

    args = parser.parse_args()
    if args.data_name == 'JD_2':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_2'
    elif args.data_name == 'JD_3':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_3'
    elif args.data_name == 'JD_4':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_4'
    elif args.data_name == 'JD_1':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_1'
    elif args.data_name == 'UB_2':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_2'
    elif args.data_name == 'UB_3':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_3'
    elif args.data_name == 'UB_4':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_4'
    else:
        raise Exception('data_name cannot be None')


    layer_embeddings_load_path = join(f'./embeddings_save/{args.data_name}',
                                      f"Layer_embeddings_at_stage_{args.stage - 1}.pth")
    embeddings_load_path = join(f'./embeddings_save/{args.data_name}', f"Embeddings_at_stage_{args.stage - 1}.pth")
    LastStage_embeddings = torch.load(layer_embeddings_load_path, map_location=torch.device('cpu'))
    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_emb_{}_lr_{}_reg_{}_{}'.format(args.model_name, args.embedding_size, args.lr, args.reg_weight, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)

    model = I_CRGCN(args, dataset, LastStage_embeddings, embeddings_load_path)
    trainer = Trainer(model, dataset, args)
    saved_dict = trainer.train_model()
    # embeddings_save_path = os.path.join(args.emb_saved_path+'JD_'+str(args.stage+1), f"Layer_embeddings_at_stage_{args.stage}.pth")
    # torch.save(saved_dict, embeddings_save_path)

    logger.info('train end total cost time: {}'.format(time.time() - start))



