#from model import CycleGan
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#from torchsummary import summary           # pip install torchsummary          #for using summary to check model summary

from train_helper import CustomDatasetLoader
from generator import Generator
from discriminator import Discriminator
from loss import GANLoss

import argparse
import itertools
import os

parser = argparse.ArgumentParser(description='Code for training CycleGan on DeHazing task')
parser.add_argument('-d','--data',help="root directory where all data for training is stored; default='data/'",required=True,default='data/')
parser.add_argument('-b','--batch-size',type=int,help='specify batch-size for training',required=True,default=8)
parser.add_argument('-n','--num-workers',help="number of DataLoader workers/threads",required=True,type=int,default=4)
args = parser.parse_args()

def train(learning_rate=0.0002, beta1=0.5,epochs=1):
    
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


    # create G, F
    print('Loading Generators(G & F)...')
    G = Generator()
    F = Generator()
    print('Generators(G & F) loaded successfully...')

    # create Dx, Dy
    print('Loading Discriminators(Dx, Dy)...')
    Dx = Discriminator()
    Dy = Discriminator()
    print('Discriminators(Dx, Dy) loaded successfully...')

    
    
    # check generator summary
    
    #summary(G,(3,256,256))
    # OR
    print(G)                                        # print Generator

    # check discriminator summary
    
    #summary(Dx,(3,256,256))
    # OR
    print(Dx)                                       # print Discriminator


    # create 3-loss_functions - Adv_loss, Cycle_consistent_loss, perceptual_loss
    criterionGAN = GANLoss().to(device)                                           ############## change device
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    # create optimizers
    optimizers = []
    optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr = learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(itertools.chain(Dx.parameters(), Dy.parameters()), lr = learning_rate, betas=(beta1, 0.999))
    optimizers.append(optimizer_G)
    optimizers.append(optimizer_D)
            
    # make dataset ready for training
    data_loader = CustomDatasetLoader()
    dataset = data_loader.load_data()
    print('Number of training images = %d' % len(dataset))

    # iterate over dataset for training
    for epoch in range(epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        #epoch_start_time = time.time()  # timer for entire epoch
        #iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, batch in enumerate(dataset):  # inner loop within one epoch
            

    


if __name__=='__main__':
    train()