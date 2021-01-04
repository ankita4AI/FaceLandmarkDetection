import random
import torch
import numpy as np
import imutils
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


class Transforms():
    def __init__(self):
        pass

    @staticmethod
    def rotate(image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    @staticmethod
    def resize(image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    @staticmethod
    def color_jitter(image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    @staticmethod
    def crop_face(image, landmarks, edges):
        left = int(edges['left'])
        top = int(edges['top'])
        width = int(edges['width'])
        height = int(edges['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, edges):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, edges)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks