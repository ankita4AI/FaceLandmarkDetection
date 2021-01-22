import torch
import torch.nn as nn
import os
import numpy as np
import random
import time
import torch.optim as optim
from FaceLandmarkDetection.src.detection.model.resnet import resnet18
from FaceLandmarkDetection.src.detection.data.dataset import FaceLandmarksDataset
from FaceLandmarkDetection.src.detection.data.prepare_data import Transforms

data_dir = 'FaceLandmarkDetection/data/face_landmark_dataset'
data_file = 'FaceLandmarkDetection/data/face_landmark_dataset/labels_ibug_300W_train.xml'


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
def create_model(num_classes=10):
    model = resnet18(pretrained=True)

    # We would use the pretrained ResNet18 as a feature extractor.
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the input channels
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the last FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_data():
    transformed_dataset = FaceLandmarksDataset(data_file=data_file, data_dir=data_dir, transform=Transforms())
    toy_factor = 1/6
    toy_dataset, other_dataset = torch.utils.data.random_split(transformed_dataset,
                                              [int(len(transformed_dataset)*toy_factor), int(len(transformed_dataset)*(1-toy_factor))])
    # split the dataset into validation and test sets
    print(len(toy_dataset))
    len_valid_set = int(0.1 * len(toy_dataset))
    len_train_set = len(toy_dataset) - len_valid_set
    train_dataset, valid_dataset = torch.utils.data.random_split(toy_dataset, [len_train_set, len_valid_set])
    return train_dataset, valid_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)
    return loader


class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

    
def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.view(labels.size(0), -1).float()
        _ = model(inputs)

        
def measure_inference_latency(model, device, input_size=(1,3,32,32), num_samples=100):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    start_time = time.time()
    for _ in range(num_samples):
        _ = model(x)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples
    return elapsed_time_ave


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

    
def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=lambda storage, loc: storage))
    return model


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

    
def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.view(labels.size(0), -1).float()
        outputs = model(inputs)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        
    eval_loss = running_loss / len(test_loader.dataset)
    return eval_loss


def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def train_model(model, train_loader, test_loader, device, num_epochs):
    learning_rate = 1e-2
    criterion = nn.MSELoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0
        # pruner.update_epoch(epoch)
        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(labels.size(0), -1).float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)
        print("Epoch: {:02d} Train Loss: {:.3f} Eval Loss: {:.3f}".format(epoch, train_loss, eval_loss))
    return model