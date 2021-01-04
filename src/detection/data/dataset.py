from torch.utils.data import Dataset
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2


class FaceLandmarksDataset(Dataset):

    def __init__(self, data_file, data_dir, transform=None):

        tree = ET.parse(data_file)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.edges = []
        self.transform = transform
        self.root_dir = data_dir

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.edges.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.edges[index])

        # zero - centre the landmarks by reducing val by 0.5 for easier learning
        landmarks = landmarks - 0.5
        return image, landmarks