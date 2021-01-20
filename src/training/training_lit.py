import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from FaceLandmarkDetection.src.detection.data.data_module import FaceLandMarkDataModule
from FaceLandmarkDetection.config import *
from FaceLandmarkDetection.src.training.lit_model import FaceLandMarkLitModel

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filepath=model_checkpoint,
    dirpath=lit_model_checkpoint_path,
    save_top_k=3,
    mode='min')

params = dict(
    epochs=num_epochs,
    num_classes=num_classes,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dataset=dataset,
    train_val_split_ratio=train_val_split_ratio,
    architecture=architecture)

wandb_logger = WandbLogger(project=project, config=params)

with wandb.init(project=project, config=params):
    # access all HPs through wandb.config, so logging matches execution!
    dm = FaceLandMarkDataModule(data_dir)
    trainer = pl.Trainer(max_epochs=wandb.config.epochs,
                         logger=wandb_logger,
                         progress_bar_refresh_rate=20,
                         checkpoint_callback=checkpoint_callback)
    model = FaceLandMarkLitModel()
    trainer.fit(model, dm)