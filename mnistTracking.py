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
        """ reshapes image inputs to a single 748 length vector """
        return self.network(x.view(-1, 784))


    def training_step(self, batch, batch_id):
        """ expects integer labels """
        images, labels = batch
        assignmentLogProbs = self.forward(images)
        return self.lossFn( assignmentLogProbs, labels )


    def test_step(self, batch, batch_id):
        """ test accuracy of labelling """
        images, labels = batch
        _, inds = torch.max( self.forward(images), 1)
        accuracy = sum( inds == labels ) / len(labels)
        self.log( 'accuracy', accuracy)
        return accuracy


    def configure_optimizers(self):
        return self.optimizer




