import torch
import torch.nn as nn
import pytorch_lightning as pl



class linearNetwork(pl.LightningModule):
    """ Basic MLP for MNIST
        Some ideas for improving performance (if necessary):
            - for training, random split with equal numbers of each label
     """
    def __init__(self):
        super(linearNetwork, self).__init__()
        self.network = nn.Sequential( 
            nn.Linear(784, 196), nn.ReLU(), 
            nn.Linear(196, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.LogSoftmax(dim=1) )

        self.lossFn = nn.NLLLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)



    def forward(self, x):
        """ Expects pixels in a list of lenght 784"""
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """ expects one  """
        images, labels = batch
        assignmentLogProbs = self.forward(images)
        return self.lossFn( assignmentLogProbs, labels )

    def configure_optimizers(self):
        return self.optimizer



