# a script to set up and run training
import pytorch_lightning as pl

from mnistTracking import linearNetwork
from mnistDataLoader import getMNISTDataLoaders

train, val, test = getMNISTDataLoaders()
model = linearNetwork()

trainer = pl.Trainer(max_epochs=30)
trainer.fit(model=model, train_dataloaders=train)
