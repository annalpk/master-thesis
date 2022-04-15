import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import re


class ViSal(data.Dataset):
    def __init__(self, dataset='test', data_dir=None, transform=None, seq_len=None, return_size=False):
        self.return_size = return_size
        self.data_dir = data_dir
        imgset_path = self.data_dir + '/ViSal-imgset.txt'
        imgset_file = open(imgset_path)
        all_of_it = imgset_file.read()
        self.seq_paths = all_of_it.split("\n\n")[:-1]

        self.folder_paths_list = []
        for x in range(len(self.seq_paths)):
            paths = self.seq_paths[x].split("\n")
            self.folder_paths_list.append(paths)

        # if no sequence legth is given, the max sequence length from all sequences is found and used
        if seq_len == None:
            self.seq_len = self.findMaxSeq()
        else:
            self.seq_len = seq_len

        images_path = []
        labels_path = []
        self.sequence_img_paths = []
        self.sequence_label_paths = []
        counter = 0
        for x in range(len(self.folder_paths_list)):
            for y in range(len(self.folder_paths_list[x])):
                print(self.folder_paths_list[x][y])
                if counter < self.seq_len:
                    splitted = self.folder_paths_list[x][y].split(" ")
                    img_path = splitted[0]
                    gt_path = splitted[1]
                    images_path.append(img_path)
                    labels_path.append(gt_path)
                    counter += 1
                else:
                    counter = 0
                    self.sequence_img_paths.append(images_path)
                    self.sequence_label_paths.append(labels_path)
                    images_path = []
                    labels_path = []

                    splitted = self.folder_paths_list[x][y].split(" ")
                    img_path = splitted[0]
                    gt_path = splitted[1]
                    images_path.append(img_path)
                    labels_path.append(gt_path)
                    counter += 1
            self.sequence_img_paths.append(images_path)
            self.sequence_label_paths.append(labels_path)
            images_path = []
            labels_path = []
            counter = 0

        self.transform = transform
        self.folder = None

    def __getitem__(self, item):
        imags = []
        labels = []
        label_names = []
        sizes = []
        counter = 0

        for x in range(len(self.sequence_img_paths[item])):
            image_path = self.data_dir + self.sequence_img_paths[item][x]
            label_path = self.data_dir + self.sequence_label_paths[item][x]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path))
            if label.max() > 0:
                label = label / label.max()

            label = Image.fromarray(label.astype(np.uint8))

            w, h = image.size
            size = (h, w)
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            if self.return_size:
                sizes.append(torch.tensor(size))

            img = sample['image']
            lab = sample['label']

            imags.append(img)
            labels.append(lab)

            pos_list = [i.start() for i in re.finditer('/', label_path)]
            label_name = label_path[pos_list[-2] + 1:]
            point_list = [i.start() for i in re.finditer('\.', label_path)]
            self.folder = label_name[:label_name.rfind("/")]
            self.number = int(label_path[pos_list[-1] + 1:point_list[0]])
            label_names.append(label_name)
            c, h, w = sample['image'].shape
            counter += 1
        if (counter >= len(self.sequence_img_paths[item]) and counter < self.seq_len):
            size = torch.tensor(size)
            for x in range(counter, self.seq_len):
                label_name = self.folder + "/{:05d}.jpg".format(self.number + x)
                image = torch.zeros((3, h, w))
                imags.append(image)
                label = torch.zeros(1, h, w)
                labels.append(label)
                sizes.append(size)
                label_names.append(label_name)
        imags = torch.stack(imags)
        labels = torch.stack(labels)

        sequence = {'images': imags, 'labels': labels, 'size': sizes, 'label_name': label_names}

        return sequence

    def __len__(self):
        return len(self.sequence_img_paths)


if __name__ == '__main__':
    print("hello world")