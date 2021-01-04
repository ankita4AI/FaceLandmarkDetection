import torch
import pytorch_lightning as pl

from FaceLandmarkDetection import config
from FaceLandmarkDetection.src.detection.model.network import Network

mse_loss = torch.nn.MSELoss()


class FaceLandMarkLitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = config.learning_rate
        self.model = Network(config.num_classes)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1).float()
        outputs = self.forward(x)
        loss = mse_loss(outputs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1).float()
        outputs = self.forward(x)
        loss = mse_loss(outputs, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer