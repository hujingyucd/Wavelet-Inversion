import torch
import os
import numpy as np
from tqdm import tqdm
from models.network import create_coordinates
from models.network_ae import VoxelAutoEncoder
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SDFSamples(torch.utils.data.Dataset):
    def __init__(self, data_files=None):
        super(SDFSamples, self).__init__()

        ## load the data folder
        self.data_preloaded = [ np.load(data_file[0]) for data_file in data_files ]
        ### data length
        self.data_len = self.data_preloaded[0].shape[0]


    def __len__(self):
        return self.data_len
        

    def __getitem__(self, idx):

        processed_data = tuple([data[idx] for data in self.data_preloaded])
            
        return processed_data, idx
