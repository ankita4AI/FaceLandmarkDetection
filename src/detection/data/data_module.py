import pytorch_lightning as pl
import torch
from FaceLandmarkDetection.src.detection.data.dataset import FaceLandmarksDataset
from FaceLandmarkDetection.src.detection.data.prepare_data import Transforms
from FaceLandmarkDetection.config import data_dir, data_file, train_val_split_ratio, batch_size


class FaceLandMarkDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = ''):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.transform = Transforms()
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.num_classes = 136

    def prepare_data(self):
        # download from S3 bucket and put into data directory
        pass

    def setup(self, stage=None):
        transformed_dataset = FaceLandmarksDataset(data_file=data_file, data_dir=data_dir, transform=self.transform)
        total_samples = len(transformed_dataset)
        toy_factor = 1/3
        toy_samples = int(toy_factor*total_samples)
        toy_dataset, other = torch.utils.data.random_split(transformed_dataset,
                                                           [toy_samples, total_samples-toy_samples])
        # split the dataset into validation and test sets
        len_valid_set = int(train_val_split_ratio * len(toy_dataset))
        len_train_set = len(toy_dataset) - len_valid_set
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(toy_dataset, [len_train_set,
                                                                                                     len_valid_set])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,  shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size)

