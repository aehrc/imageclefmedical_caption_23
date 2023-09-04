import glob
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageCLEFSubset(Dataset):
    def __init__(self, df, transforms, images_dir, colour_space='RGB'):
        super().__init__()
    
        self.df = df
        self.transforms = transforms
        self.images_dir = images_dir
        self.colour_space = colour_space

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, index):
        example = self.df.iloc[index].to_dict()
        image = Image.open(os.path.join(self.images_dir, example['ID'] + '.jpg'))
        image = image.convert(self.colour_space)
        if self.transforms is not None:
            image = self.transforms(image)
        return {'images': image, 'captions': example['caption'], 'ids': example['ID']}

class ImageCLEFTestSet(Dataset):
    def __init__(self, transforms, images_dir, colour_space='RGB'):
        super().__init__()
    
        self.transforms = transforms
        self.images_dir = images_dir
        self.colour_space = colour_space

        self.image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))

    def __len__(self):
        return self.image_paths.__len__()

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        image = image.convert(self.colour_space)
        if self.transforms is not None:
            image = self.transforms(image)

        return {'images': image, 'ids': Path(self.image_paths[index]).stem}