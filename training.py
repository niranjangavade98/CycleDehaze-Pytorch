import torch
from model import CycleGan

import argparse
import os

parser = argparse.ArgumentParser(description='Code for training CycleGan on DeHazing task')
parser.add_argument('-d','--data',help="root directory where all data for training is stored; default='data/'",required=True,default='data/')
parser.add_argument('-b','--batch-size',type=int,help='specify batch-size for training',required=true,default=8)
args = parser.parse_args()

def train():
    # parse data from args passed
    data_dir = args.data
    batch = args.batch_size

    assert os.path.isdir(data_dir), "{} is not a valid directory".format(data_dir) #check if data dir exists
    
    # create G, F

    # create D_Y, D_X

    # create 3-loss_functions - Adv_loss, Cycle_consistent_loss, perceptual_loss

    # create transforms for data

    # create custom DataLoader

    # iterate over the dataset to train


if __name__=='__main__':
    train()