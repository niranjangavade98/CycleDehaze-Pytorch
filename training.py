from torch.utils.data import Dataset
from model import CycleGan

import argparse
import os
import random

parser = argparse.ArgumentParser(description='Code for training CycleGan on DeHazing task')
parser.add_argument('-d','--data',help="root directory where all data for training is stored; default='data/'",required=True,default='data/')
parser.add_argument('-b','--batch-size',type=int,help='specify batch-size for training',required=true,default=8)
args = parser.parse_args()

class DehazeDataset(Dataset):
    "Dehazing Dataset"
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.train_X='train_X'
        self.train_Y='train_Y'
        self.X_dir_list = os.listdir(os.path.join(self.root_dir, self.train_X))
        self.Y_dir_list = os.listdir(os.path.join(self.root_dir, self.train_Y))
        random.shuffle(self.X_dir_list)
        random.shuffle(self.Y_dir_list)
        self.iter=0
    def __len__(self):
        assert len(self.X_dir_list) == len(self.Y_dir_list), "Number of files in {} & {} are not the same".format(self.train_X,self.train_Y)
        return len(self.X_dir_list)
    def __getitem__(self,idx):
        pass
    def get_batch(self, batch_size=7):
        """
        Get a batch of image names from train_X & train_Y of specified batch_size

        Args:
            batch_size : size of batch to return
        """
        assert len(self.X_dir_list) == len(self.Y_dir_list), "Number of files in {} & {} are not the same".format(self.train_X,self.train_Y)
        if self.iter < len(self.X_dir_list):
            if self.iter+batch_size >= len(self.X_dir_list):
                batch_X = self.X_dir_list[self.iter:]
                batch_Y = self.Y_dir_list[self.iter:]
            else:
                batch_X = self.X_dir_list[self.iter:self.iter+batch_size]
                batch_Y = self.Y_dir_list[self.iter:self.iter+batch_size]
            self.iter=self.iter+batch_size
            return batch_X, batch_Y
        else:
            print("All samples from the dataset have been used; please reset the dataset by using <reset_dataset> func")
    def reset_dataset(self):
        """
        Re-shufles train_X & train_Y & sets self.iter to 0
        """
        self.iter=0
        random.shuffle(self.X_dir_list)
        random.shuffle(self.Y_dir_list)


def train():
    # parse data from args passed
    data_dir = args.data
    batch = args.batch_size

    #check if data dir exists
    assert os.path.isdir(data_dir), "{} is not a valid directory".format(data_dir)
    
    # create transforms for data

    # create custom DataLoader

    # create G, F

    # create D_Y, D_X

    # create 3-loss_functions - Adv_loss, Cycle_consistent_loss, perceptual_loss

    # iterate over the dataset to train


if __name__=='__main__':
    train()