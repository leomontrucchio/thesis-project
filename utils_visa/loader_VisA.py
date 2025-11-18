import os
import torch
from PIL import Image, ImageOps
from torchvision import transforms
import glob
from torch.utils.data import Dataset, DataLoader
from utils_visa.general_utils import SquarePad


class DeepSeekPad:
    def __init__(self, size, fill_color=(127, 127, 127)):
        self.target_size = (size, size)
        self.fill_color = fill_color

    def __call__(self, pil_img):
        return ImageOps.pad(pil_img, self.target_size, color=self.fill_color)

class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        self.SIGLIP_MEAN = [0.5, 0.5, 0.5]
        self.SIGLIP_STD = [0.5, 0.5, 0.5]

        self.size = img_size
        self.img_path = os.path.join(dataset_path, class_name, split)

        fill_color_rgb = tuple(int(x * 255) for x in self.SIGLIP_MEAN)

        self.rgb_transform = transforms.Compose([
            DeepSeekPad(self.size, fill_color=fill_color_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.SIGLIP_MEAN, std=self.SIGLIP_STD)
        ])

class TrainValDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(split = split, class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths = self.load_dataset()

    def load_dataset(self):
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.JPG")
        rgb_paths.sort()
        return rgb_paths

    def __len__(self):
        return len(self.img_paths)

    def get_size(self):
        return max(Image.open(self.img_paths[0]).convert('RGB').size)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        pil_img = Image.open(rgb_path).convert('RGB')
        tensor_img = self.rgb_transform(pil_img)

        return pil_img, tensor_img

class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split = "test", class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                rgb_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend([0] * len(rgb_paths))
                tot_labels.extend([0] * len(rgb_paths))
            elif defect_type == 'bad':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
                rgb_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(rgb_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def get_size(self):
        return max(Image.open(self.img_paths[0]).convert('RGB').size)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]

        pil_img = Image.open(img_path).convert('RGB')

        if gt == 0:
            gt = torch.zeros(
                [1, pil_img.size[1], pil_img.size[0]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        tensor_img = self.rgb_transform(pil_img)

        return pil_img, tensor_img, gt, label, img_path


def custom_collate_fn(batch):

    num_items = len(batch[0])

    if num_items == 2:  # Training
        pil_images = [item[0] for item in batch]
        tensor_images = torch.stack([item[1] for item in batch], 0)
        return pil_images, tensor_images

    elif num_items == 5:  # Test
        pil_images = [batch[0][0]]
        tensor_images = batch[0][1].unsqueeze(0)
        gts = batch[0][2].unsqueeze(0)
        labels = torch.tensor([batch[0][3]])
        img_paths = [batch[0][4]]

        return pil_images, tensor_images, gts, labels, img_paths


def get_data_loader(split, class_name, dataset_path, img_size, batch_size=None):
    if split in ['train']:
        dataset = TrainValDataset(split = "train", class_name = class_name, img_size = img_size, dataset_path = dataset_path)
        data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2, drop_last = False, pin_memory = False, collate_fn=custom_collate_fn)
    elif split in ['test']:
        dataset = TestDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
        data_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False, num_workers = 2, drop_last = False, pin_memory = False, collate_fn=custom_collate_fn)

    return data_loader, dataset.get_size()