#from model import CycleGan
from torch.utils.data import DataLoader
#from torchsummary import summary           # pip install torchsummary          #for using summary to check model summary

from train_helper import DehazeDataset
from generator import Generator
from discriminator import Discriminator

import argparse
import os

parser = argparse.ArgumentParser(description='Code for training CycleGan on DeHazing task')
parser.add_argument('-d','--data',help="root directory where all data for training is stored; default='data/'",required=True,default='data/')
parser.add_argument('-b','--batch-size',type=int,help='specify batch-size for training',required=True,default=8)
parser.add_argument('-n','--num-workers',help="number of DataLoader workers/threads",required=True,type=int,default=4)
args = parser.parse_args()

def test():
    pass