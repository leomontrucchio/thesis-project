import argparse
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile

from utils_mvtec_loco.loader_loco import get_data_loader
from utils_visa.general_utils import set_seeds

from models.teacher import ViTFeatureExtractor
from models.student import ResidualFeatureProjectionMLP



def infer(args):

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloader.
    test_loader, max_hw = get_data_loader(
        "test", class_name = args.class_name,
        img_size = args.img_size,
        dataset_path = args.dataset_path)

    # Feature extractor.
    fe = ViTFeatureExtractor(layers = [1,5]).to(device).eval()

    # Model instantiation.
    backward_net = ResidualFeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)
    forward_net = ResidualFeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)

    forward_net_path = rf'{args.checkpoint_folder}/{args.class_name}/forward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    backward_net_path = rf'{args.checkpoint_folder}/{args.class_name}/backward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'

    forward_net.load_state_dict(torch.load(forward_net_path, weights_only=False))
    backward_net.load_state_dict(torch.load(backward_net_path, weights_only=False))

    # Send students to GPU.
    forward_net.to(device)
    backward_net.to(device)

    # Make students non-trainable.
    forward_net.eval()
    backward_net.eval()

    # Gaussian blur parameters.
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device)/(w_u**2)

    # Base folder to store anomaly maps
    base_output_dir = os.path.join(args.anomaly_maps_dir, args.class_name, 'test')

    for pil_img, tensor_img, img_path_list in tqdm(test_loader, desc = f'Extracting features from class: {args.class_name}.'):

        img_path = img_path_list[0]

        with torch.no_grad():

            tensor_img = tensor_img.to(device)

            # Feature extraction.
            middle_patch, last_patch = fe(tensor_img)

            # Nets prediction.
            predicted_middle_patch = backward_net(last_patch)
            predicted_last_patch = forward_net(middle_patch)

            middle_anomaly_map = (torch.nn.functional.normalize(predicted_middle_patch, dim = -1) - torch.nn.functional.normalize(middle_patch, dim = -1)).pow(2).sum(-1).sqrt()
            last_anomaly_map = (torch.nn.functional.normalize(predicted_last_patch, dim = -1) - torch.nn.functional.normalize(last_patch, dim = -1)).pow(2).sum(-1).sqrt()

            combined_anomaly_map = (middle_anomaly_map * last_anomaly_map).reshape(1, 1, args.img_size // fe.patch_size, args.img_size // fe.patch_size)

            # Upsample to original resolution.
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size = [max_hw, max_hw], mode = 'bilinear')

            # Approximated Gaussian blur.
            for _ in range(5):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_u, weight=weight_u)

            combined_anomaly_map = combined_anomaly_map.reshape(max_hw, max_hw)

            original_h, original_w = pil_img[0].size[1], pil_img[0].size[0]
            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, [original_h, original_w])

            parent_dir = os.path.basename(os.path.dirname(img_path)) # ex. logical_anomalies
            file_name = os.path.basename(img_path) # ex. 000.png
            file_name_no_ext = os.path.splitext(file_name)[0] # ex. 000

            save_dir = os.path.join(base_output_dir, parent_dir)
            save_path = os.path.join(save_dir, f"{file_name_no_ext}.tiff")

            os.makedirs(save_dir, exist_ok=True)

            map_numpy = combined_anomaly_map.cpu().numpy().astype(np.float32)
            tifffile.imwrite(save_path, map_numpy)
    
    print("Inference complete. Maps are stored.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Inference framework')

    parser.add_argument('--checkpoint_folder', default = './checkpoints/checkpoints_loco', type = str,
                        help = 'Path to the folder containing students checkpoints.')

    parser.add_argument('--anomaly_maps_dir', default='./results/loco/anomaly_maps', type=str,
                        help='Base directory where anomaly maps will be saved.')

    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str,
                        help = 'Dataset path.')

    parser.add_argument('--class_name', default = "breakfast_box", type = str,
                        help = 'Category name.')

    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')

    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--label', default = 'l2bt_deepseek_res_new', type = str, 
                        help = 'Label to identify the experiment.')

    args = parser.parse_args()
    infer(args)