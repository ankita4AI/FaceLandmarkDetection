from FaceLandmarkDetection.config import data_file, data_dir, project, save_model_dir, num_epochs, num_classes, \
    batch_size, learning_rate, dataset, train_val_split_ratio, architecture
from FaceLandmarkDetection.src.detection.data.dataset import FaceLandmarksDataset
from FaceLandmarkDetection.src.detection.data.prepare_data import Transforms
from FaceLandmarkDetection.src.detection.model.network import Network
import copy
import torch
import numpy as np
import wandb
import random


def make(config):
    # Make the data
    train, val = get_data(config=config)
    train_loader = make_loader(train, batch_size=config.batch_size)
    val_loader = make_loader(val, batch_size=config.batch_size)

    # Make the model
    model = Network(config.classes).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return model, train_loader, val_loader, criterion, optimizer, scheduler


def get_data(config=None):
    transformed_dataset = FaceLandmarksDataset(data_file=data_file, data_dir=data_dir, transform=Transforms())
    # creating toy dataset (sampled)
    toy_dataset, other_dataset = torch.data.utils.random_split(transformed_dataset,
                                              [len(transformed_dataset) // 6, 5 * len(transformed_dataset) // 6])
    # split the dataset into validation and test sets
    len_valid_set = int(config.train_val_split_ratio * len(toy_dataset))
    len_train_set = len(toy_dataset) - len_valid_set
    train_dataset, valid_dataset = torch.utils.data.random_split(toy_dataset, [len_train_set, len_valid_set])
    return train_dataset, valid_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)
    return loader


def train_model(model, dataloaders, criterion, optimizer, scheduler, config):

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss =np.inf

    for epoch in range(config.epochs):
        print('Epoch {}/{}'.format(epoch, config.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            example_ct = 0  # number of examples seen
            batch_ct = 0

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                example_ct += len(inputs)
                batch_ct += 1

                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(labels.size(0), -1).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if ((batch_ct + 1) % 25) == 0:
                    process_log(loss, example_ct, epoch, phase)

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def process_log(loss, example_ct, epoch, process):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(process + f" Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def save_model(model, save_model_dir):
    torch.save(model.state_dict(), save_model_dir)


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project=project, config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, val_loader, criterion, optimizer, scheduler = make(config)
        print(model)
        wandb.watch(model)
        dataloaders = {"train": train_loader, "val": val_loader}
        # and use them to train the model
        model = train_model(model, dataloaders, criterion, optimizer, scheduler, config)

        # save the model
        save_model(model, save_model_dir)
    return model


if __name__ == "main":
    # LogIn to wandb
    wandb.login()
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict(
        epochs=num_epochs,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset=dataset,
        train_val_split_ratio=train_val_split_ratio,
        architecture=architecture)

    model = model_pipeline(params)