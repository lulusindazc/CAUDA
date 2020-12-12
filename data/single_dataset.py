import os
from .image_folder import make_dataset_with_labels, make_dataset
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.data_labels[index] 

        return {'Path': path, 'Img': img, 'Label': label,'index':index}

    def initialize(self, root, transform=None, **kwargs):
        self.root = root
        self.data_paths = []
        self.data_labels = []
        self.transform = transform
    def initialize_path(self, path,label, transform=None, **kwargs):
        # self.root = root
        self.data_paths = path
        self.data_labels = label
        # self.transform = transform

    def __len__(self):
        return len(self.data_paths)

# class SingleDataset(BaseDataset):
#     def initialize(self, root, classnames, transform=None, **kwargs):
#         BaseDataset.initialize(self, root, transform)
#         self.data_paths, self.data_labels = make_dataset_with_labels(
# 				self.root, classnames)
#
#         assert(len(self.data_paths) == len(self.data_labels)), \
#             'The number of images (%d) should be equal to the number of labels (%d).' % \
#             (len(self.data_paths), len(self.data_labels))
#
#     def initialize_path(self, path,label, transform=None, **kwargs):
#         BaseDataset.initialize_path(self, path,label, transform)
#         # self.data_paths, self.data_labels = make_dataset_with_labels(
#         #     self.root, classnames)
#         self.data_paths, self.data_labels =path,label
#         assert (len(self.data_paths) == len(self.data_labels)), \
#             'The number of images (%d) should be equal to the number of labels (%d).' % \
#             (len(self.data_paths), len(self.data_labels))
#     def name(self):
#         return 'SingleDataset'

class BaseDatasetWithoutLabel(Dataset):
    def __init__(self):
        super(BaseDatasetWithoutLabel, self).__init__()

    def name(self):
        return 'BaseDatasetWithoutLabel'

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return {'Path': path, 'Img': img,'index':index}

    def initialize(self, root, transform=None, **kwargs):
        self.root = root
        self.data_paths = []
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)
#
# class SingleDatasetWithoutLabel(BaseDatasetWithoutLabel):
#     def initialize(self, root, transform=None, **kwargs):
#         BaseDatasetWithoutLabel.initialize(self, root, transform)
#         self.data_paths = make_dataset(self.root)
#
#     def name(self):
#         return 'SingleDatasetWithoutLabel'


