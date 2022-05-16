# a script to setup training
import pytorch_lightning as pl

from mnistTracking import linearNetwork
from mnistDatasets import getMNISTDatasets

from torch.utils.data import DataLoader



if __name__ == '__main__':
    # make dataloaders
    trainData, validateData, testData = getMNISTDatasets()

    trainDL = DataLoader(trainData, batch_size=10, shuffle=True, num_workers=4)
    testDL = DataLoader(testData, batch_size=len(testData))

    # initialize the model
    model = linearNetwork()

    trainer = pl.Trainer(max_epochs=30)

    # run these for trianing and testing
    #trainer.fit(model, trainDL)
    #trainer.test(model, testDL)