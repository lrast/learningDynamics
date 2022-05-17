# a script to setup training
import pytorch_lightning as pl

from mnistTracking import linearNetwork
from mnistDatasets import getMNISTDatasets

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    #data
    datasets = getMNISTDatasets()
    trainDL = DataLoader(datasets[0], batch_size=10, shuffle=True, num_workers=4)
    testDL = DataLoader(datasets[2], batch_size=len(datasets[2]))

    # model
    model = linearNetwork()

    # tracking
    callbacks = [ModelCheckpoint(every_n_epochs=1, save_weights_only=True, save_top_k=-1)]

    trainer = pl.Trainer(max_epochs=30, callbacks=callbacks)

    # run these for training and testing
    #trainer.fit(model, trainDL)
    #trainer.test(model, testDL)