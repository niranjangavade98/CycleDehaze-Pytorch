#from model import CycleGan
from torch.utils.data import DataLoader

from train_helper import DehazeDataset

import argparse
import os

parser = argparse.ArgumentParser(description='Code for training CycleGan on DeHazing task')
parser.add_argument('-d','--data',help="root directory where all data for training is stored; default='data/'",required=True,default='data/')
parser.add_argument('-b','--batch-size',type=int,help='specify batch-size for training',required=True,default=8)
parser.add_argument('-n','--num-workers',help="number of DataLoader workers/threads",required=True,type=int,default=4)
args = parser.parse_args()

def train():
    
    # parse data from args passed
    data_dir = args.data
    batch_size = args.batch_size
    num_workers = args.num_workers

    #check if data dir exists
    assert os.path.isdir(data_dir), "{} is not a valid directory".format(data_dir)

    # create dataset (transforms are also included in this only)
    print('Loading dataset...')
    dataset = DehazeDataset(data_dir)
    print('Dataset loaded successfully...')
    print('Dataset contains {} distinct datapoints in X(source) & Y(target) domain\n\n'.format(len(dataset)))

    # create custom DataLoader
    dataloader = DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

    print(dataset[0])


    # create G, F

    # create D_Y, D_X

    # create 3-loss_functions - Adv_loss, Cycle_consistent_loss, perceptual_loss

    # iterate over the dataset to train


if __name__=='__main__':
    train()