from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
from PIL import Image
import random

class DehazeDataset(Dataset):
    "Dehazing Dataset"
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.train_X='train_X'
        self.train_Y='train_Y'
        self.X_dir_list = os.listdir(os.path.join(self.root_dir, self.train_X))     # get list of image paths in domain X
        self.Y_dir_list = os.listdir(os.path.join(self.root_dir, self.train_Y))     # get list of image paths in domain Y
        random.shuffle(self.X_dir_list)
        random.shuffle(self.Y_dir_list)
        self.transforms = self.get_transforms()                                     # get transforms to apply to all images


    def __len__(self):
        assert len(self.X_dir_list) == len(self.Y_dir_list), "Number of files in {} & {} are not the same".format(self.train_X,self.train_Y)
        return len(self.X_dir_list)

    def __getitem__(self,index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains X, Y, X_paths, Y_paths
            X (tensor)       -- an image in the input domain
            Y (tensor)       -- image in the target (randomly chosen)
            X_paths (str)    -- image paths
            Y_paths (str)    -- image paths
        """
        X_img_path = self.X_dir_list[index % len(self.X_dir_list)]  # make sure index is within the range
        # randomize the index for domain Y to avoid fixed pairs.
        ind_Y = random.randint(0, len(self.Y_dir_list) - 1)
        Y_img_path = self.Y_dir_list[ind_Y]
        X_img = Image.open(os.path.join(self.root_dir,self.train_X,X_img_path)).convert('RGB')
        Y_img = Image.open(os.path.join(self.root_dir,self.train_Y,Y_img_path)).convert('RGB')
        # apply image transformation
        X = self.transforms(X_img)
        Y = self.transforms(Y_img)
        return {'X': X, 'Y': Y, 'X_paths': X_img_path, 'Y_paths': Y_img_path}

    def get_transforms(self, resize_to=286, interpolation=Image.BICUBIC, crop_size=256):
        """
        Returns 'transforms.Compose object' to apply on images.   Applies resize,randomCrop,
                                                                          randomHorizFlip,toTensor,
                                                                          Normalise
        Args:
            resize_to     : size to resize images
            interpolation : method to resize with
            crop_size     : size to crop images
        """
        all_transforms=[]
        all_transforms.append(transforms.Resize(size=(resize_to,resize_to), interpolation=interpolation))
        all_transforms.append(transforms.RandomCrop(crop_size))
        all_transforms.append(transforms.RandomHorizontalFlip())
        all_transforms.append(transforms.ToTensor())
        all_transforms.append(transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ))
        return transforms.Compose(all_transforms)
    
    '''
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
    '''

class CustomDatasetLoader():

    def __init__(self, root_dir='./data', batch_size=1, threads=1):
        self.dataset = DehazeDataset(root_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=int(threads))

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
            
    def load_data(self):
        return self
