import torch
import torchvision

from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype, Normalize, Lambda
from torch.utils.data import DataLoader, random_split

def getMNISTDatasets(split=[0.6, 0.2, 0.2], batch=10):
    """ Return train, validate, and test random splits of MNIST
    Downloads the data if it is not present"""

    # preprocess
    MNISTdataset = torchvision.datasets.MNIST('~/Datasets/MNIST', 
        transform= Compose( [PILToTensor(), 
                            ConvertImageDtype(torch.float), 
                            Normalize(0.,1.)]),
        download=False) # toggle to get the dataset

    # split
    total = sum(split)
    sizes = [ int(60000 * frac / total ) for frac in split ]

    trainDataset, validateDataset, testDataset = random_split(MNISTdataset, sizes)
    return trainDataset, validateDataset, testDataset

