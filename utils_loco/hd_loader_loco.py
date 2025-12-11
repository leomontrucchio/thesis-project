import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --- Dataset ---
class HighResDataset(Dataset):
    def __init__(self, split, class_name, dataset_path):
        self.split = split
        self.dataset_path = dataset_path
        self.class_name = class_name
        
        # Prepare file paths
        base_path = os.path.join(dataset_path, class_name)
        if split == 'train':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'train', 'good', '*.png')))
        elif split == 'validation':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'validation', 'good', '*.png')))
        elif split == 'test':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'test', '**', '*.png'), recursive=True))
            self.img_paths = [p for p in self.img_paths if os.path.isfile(p)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        pil_img = Image.open(img_path).convert('RGB')
        
        if self.split == 'test':
            return pil_img, img_path
        else:
            return pil_img

# --- Collate Function ---
def hd_collate_fn(batch):
    # If batch contains tuples (pil, path): Test Mode
    if isinstance(batch[0], tuple): 
        pil_images = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        return pil_images, paths
    else:
        # Train/Val Mode
        return batch

# --- Get Data Loader ---
def get_hd_data_loader(split, class_name, dataset_path, batch_size=4):

    dataset = HighResDataset(split, class_name, dataset_path)
    
    should_shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=4,
        drop_last=False,
        pin_memory = False,
        collate_fn=hd_collate_fn
    )
    
    return loader