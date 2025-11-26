import torch
import numpy as np
from torchvision import transforms


def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


# Helper function to extract metrics safely
def extract_metrics(category_key, metrics):

        i_auroc = metrics['classification']['auc_roc'].get(category_key, 0.0)
        
        spro_dict = metrics['localization']['auc_spro'].get(category_key, {})
        spro_30 = spro_dict.get('0.3', 0.0)
        spro_10 = spro_dict.get('0.1', 0.0)
        spro_05 = spro_dict.get('0.05', 0.0)
        spro_01 = spro_dict.get('0.01', 0.0)
        
        return i_auroc, spro_30, spro_10, spro_05, spro_01


siglip_denormalize = transforms.Compose([
    transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.])
    ])