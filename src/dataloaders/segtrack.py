import os
from PIL import Image
import torch
from torch.utils import data

import numpy as np
import re
import random

from . import image_transforms as trforms

class SegTrack(data.Dataset):
    def __init__(self, dataset='train', data_dir=None, transform=None, seq_len=None, return_size=False):
        self.return_size = return_size
        self.data_dir = data_dir
        self.dataset=dataset
        imgset_path = self.data_dir + '/ImageSets/train.txt'
        imgset_file = open(imgset_path)
        all_of_it = imgset_file.read()
        self.seq_paths = all_of_it.split("\n\n")[:-1]

        self.folder_paths_list = []
        for x in range(len(self.seq_paths)):
            paths = self.seq_paths[x].split("\n")
            self.folder_paths_list.append(paths)


        #if no sequence legth is given, the max sequence length from all sequences is found and used
        if seq_len == None:
            self.seq_len = self.findMaxSeq()
        else:
            self.seq_len = seq_len

        for seq in self.folder_paths_list:
            while (len(seq) % self.seq_len != 0):
                del seq[-1]

        images_path = []
        labels_path = []
        self.sequence_img_paths = []
        self.sequence_label_paths = []

        if self.dataset == 'train':
            self.sample_videos(self.sequence_img_paths, self.sequence_label_paths)

        counter = 0
        for x in range(len(self.folder_paths_list)):
            for y in range(len(self.folder_paths_list[x])):
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
        self.randomHFlip = trforms.RandomHorizontalFlip()

    def __getitem__(self, item):
        imags = []
        labels = []
        label_names = []
        sizes = []
        counter = 0
        flip = random.random()

        for x in range(len(self.sequence_img_paths[item])):
            image_path = self.data_dir + self.sequence_img_paths[item][x]
            label_path = self.data_dir + self.sequence_label_paths[item][x]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path))
            label = label[:, :, 0]
            if label.max() > 0:
                label = label / label.max()

            label = Image.fromarray(label.astype(np.uint8))

            w, h = image.size
            size = (h, w)
            sample = {'image': image, 'label': label}
            if self.dataset == "train":
                sample = self.randomHFlip(sample, flip)
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
            label_names.append(label_name)
            c, h, w = sample['image'].shape
        imags = torch.stack(imags)
        labels = torch.stack(labels)

        sequence = {'images': imags, 'labels': labels, 'size': sizes, 'label_name': label_names}

        return sequence

    def __len__(self):
        return len(self.sequence_img_paths)

    def findMaxSeq(self):
        max_number = None
        for x in range(len(self.sequence_img_paths)):
            if (max_number is None or len(self.sequence_img_paths[x]) > max_number):
                max_number = len(self.sequence_img_paths[x])

        return max_number

    def sample_videos(self, sequence_img_paths, sequence_label_paths):
        images_path = []
        labels_path = []
        for x in range(len(self.folder_paths_list)):
            for y in range(len(self.folder_paths_list[x])):
                if (y % 20 == 0):
                    for sample in range(3, 6):
                        last_frame = y + (sample * self.seq_len) - 1
                        if (last_frame < len(self.folder_paths_list[x])):
                            splitted = self.folder_paths_list[x][y].split(" ")
                            img_path = splitted[0]
                            gt_path = splitted[1]
                            images_path.append(img_path)
                            labels_path.append(gt_path)
                            frame = y
                            for num in range(self.seq_len - 1):
                                frame += sample
                                splitted = self.folder_paths_list[x][frame].split(" ")
                                img_path = splitted[0]
                                gt_path = splitted[1]
                                images_path.append(img_path)
                                labels_path.append(gt_path)
                            sequence_img_paths.append(images_path)
                            sequence_label_paths.append(labels_path)
                            images_path = []
                            labels_path = []

    def getZeroElements(self, shape, size, label_name):
        image = torch.zeros(shape)
        sample = {'image': image, 'label': image, 'size': size, 'label_name': label_name}
        return sample


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


if __name__ == '__main__':
    print("hello world")