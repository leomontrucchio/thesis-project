import argparse
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import tifffile

from utils_loco.loader_loco import get_data_loader
from utils_visa.general_utils import set_seeds

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def infer(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float32 if args.students_blocks == 'Both ViT' else torch.bfloat16

    vit_label = args.vit_label
    llm_label = args.label

    # Dataloader
    test_loader, max_hw = get_data_loader(
        "test", class_name=args.class_name,
        img_size=args.img_size, dataset_path=args.dataset_path
    )

    # Model Initialization
    students = {}
    teachers = {}
    check_dir = f'{args.checkpoint_folder}/{args.class_name}'

    if args.students_blocks == 'Both ViT':
        # --- ViT Bidirectional ---
        teachers['fe'] = ViTFeatureExtractor(layers=[1, 5]).to(device).eval()
        
        students['backward_net'] = ResidualFeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim, reduction_factor=1).to(device)
        students['forward_net'] = ResidualFeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim, reduction_factor=1).to(device)

        fwd_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        
        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Both LLM':
        # --- LLM Bidirectional ---
        teachers['fe'] = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()
        
        students['backward_net'] = FeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device=device, dtype=dtype)

        fwd_path = os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Full':
        # --- ViT Bidirectional + LLM Bidirectional ---
        teachers['vit_fe'] = ViTFeatureExtractor(layers=[1, 5]).to(device, dtype=dtype).eval()
        teachers['llm_fe'] = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()

        # ViT Students (Residual)
        students['vit_fw'] = ResidualFeatureProjectionMLP(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, reduction_factor=1).to(device=device, dtype=dtype)
        students['vit_bw'] = ResidualFeatureProjectionMLP(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, reduction_factor=1).to(device=device, dtype=dtype)
        
        # LLM Students (Standard)
        students['llm_fw'] = FeatureProjectionMLP(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim).to(device=device, dtype=dtype)
        students['llm_bw'] = FeatureProjectionMLP(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim).to(device=device, dtype=dtype)

        vit_fw_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        vit_bw_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_fw_path = os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_bw_path = os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['vit_fw'].load_state_dict(torch.load(vit_fw_path, map_location=device, weights_only=False))
        students['vit_bw'].load_state_dict(torch.load(vit_bw_path, map_location=device, weights_only=False))
        students['llm_fw'].load_state_dict(torch.load(llm_fw_path, map_location=device, weights_only=False))
        students['llm_bw'].load_state_dict(torch.load(llm_bw_path, map_location=device, weights_only=False))

    else:
        raise ValueError("Invalid students_blocks option.")

    # Set all students to eval
    for model in students.values():
        model.eval()

    # Gaussian Blur Setup
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device, dtype=dtype)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device, dtype=dtype)/(w_u**2)

    # Base folder to store anomaly maps
    base_output_dir = os.path.join(args.anomaly_maps_dir, args.class_name, 'test')

    # --- Inference Loop ---
    for pil_img, tensor_img, img_path_list in tqdm(test_loader, desc=f'Inferencing {args.class_name}'):
        
        img_path = img_path_list[0]
        
        with torch.no_grad():
            tensor_img = tensor_img.to(device, dtype=dtype)
            
            # Feature Extraction & Prediction & Map Calculation
            combined_anomaly_map = None

            if args.students_blocks == 'Both ViT':
                # Extraction
                first_patch, second_patch = teachers['fe'](tensor_img)
                # Prediction
                pred_1st_patch = students['backward_net'](second_patch)
                pred_2nd_patch = students['forward_net'](first_patch)
                # Map Calc
                first_map = (torch.nn.functional.normalize(pred_1st_patch, dim=-1) - torch.nn.functional.normalize(first_patch, dim=-1)).pow(2).sum(-1).sqrt()
                second_map = (torch.nn.functional.normalize(pred_2nd_patch, dim=-1) - torch.nn.functional.normalize(second_patch, dim=-1)).pow(2).sum(-1).sqrt()
                
                combined_anomaly_map = (first_map * second_map).reshape(1, 1, args.img_size // teachers['fe'].patch_size, args.img_size // teachers['fe'].patch_size)

            elif args.students_blocks == 'Both LLM':
                # Extraction
                first_patch, second_patch = teachers['fe'](pil_img)
                # Prediction
                pred_1st_patch = students['backward_net'](second_patch)
                pred_2nd_patch = students['forward_net'](first_patch)
                # Map Calc
                first_map = (torch.nn.functional.normalize(pred_1st_patch, dim=-1) - torch.nn.functional.normalize(first_patch, dim=-1)).pow(2).sum(-1).sqrt()
                second_map = (torch.nn.functional.normalize(pred_2nd_patch, dim=-1) - torch.nn.functional.normalize(second_patch, dim=-1)).pow(2).sum(-1).sqrt()
                
                combined_anomaly_map = (first_map * second_map).reshape(1, 1, 14, 14)

            elif args.students_blocks == 'Full':
                # Extraction
                vit_1st, vit_2nd = teachers['vit_fe'](tensor_img)
                llm_1st, llm_2nd = teachers['llm_fe'](pil_img)
                
                # Prediction
                pred_vit_2nd = students['vit_fw'](vit_1st)
                pred_vit_1st = students['vit_bw'](vit_2nd)
                pred_llm_2nd = students['llm_fw'](llm_1st)
                pred_llm_1st = students['llm_bw'](llm_2nd)

                # Map Calc
                vit_fw_map = (torch.nn.functional.normalize(pred_vit_2nd, dim=-1) - torch.nn.functional.normalize(vit_2nd, dim=-1)).pow(2).sum(-1).sqrt()
                vit_bw_map = (torch.nn.functional.normalize(pred_vit_1st, dim=-1) - torch.nn.functional.normalize(vit_1st, dim=-1)).pow(2).sum(-1).sqrt()
                llm_fw_map = (torch.nn.functional.normalize(pred_llm_2nd, dim=-1) - torch.nn.functional.normalize(llm_2nd, dim=-1)).pow(2).sum(-1).sqrt()
                llm_bw_map = (torch.nn.functional.normalize(pred_llm_1st, dim=-1) - torch.nn.functional.normalize(llm_1st, dim=-1)).pow(2).sum(-1).sqrt()

                vit_comb = (vit_fw_map.reshape(1, 1, 27, 27) * vit_bw_map.reshape(1, 1, 27, 27))
                
                llm_fw_res = torch.nn.functional.interpolate(llm_fw_map.reshape(1, 1, 14, 14), size=(27, 27), mode='bilinear', align_corners=False)
                llm_bw_res = torch.nn.functional.interpolate(llm_bw_map.reshape(1, 1, 14, 14), size=(27, 27), mode='bilinear', align_corners=False)
                llm_comb = llm_fw_res * llm_bw_res
                
                combined_anomaly_map = vit_comb * llm_comb

            # Upsample to original resolution
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=[max_hw, max_hw], mode='bilinear')

            # Approximated Gaussian blur
            for _ in range(5):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_u, weight=weight_u)

            combined_anomaly_map = combined_anomaly_map.reshape(max_hw, max_hw)

            # Center Crop to Original Size
            original_h, original_w = pil_img[0].size[1], pil_img[0].size[0]
            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, [original_h, original_w])

            # Save Map
            parent_dir = os.path.basename(os.path.dirname(img_path)) # ex. logical_anomalies
            file_name = os.path.basename(img_path) # ex. 000.png
            file_name_no_ext = os.path.splitext(file_name)[0] # ex. 000

            save_dir = os.path.join(base_output_dir, parent_dir)
            save_path = os.path.join(save_dir, f"{file_name_no_ext}.tiff")

            os.makedirs(save_dir, exist_ok=True)

            map_numpy = combined_anomaly_map.to(torch.float32).cpu().detach().numpy()
            tifffile.imwrite(save_path, map_numpy)

    print("Inference complete. Maps are stored.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference Framework')

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

    parser.add_argument('--students_blocks', type=str, default='Full', choices=['Both ViT', 'Both LLM', 'Full'],
                        help='Inference scenario.')
    
    parser.add_argument('--label', default=None, type=str,
                        help='Experiment label. For experiment involving just ViT students, use the same name of vit_label.')
    
    parser.add_argument('--vit_label', default='l2bt_deepseek_res_new', type=str,
                        help='Label of experiment involving just ViT students.')

    args = parser.parse_args()
    infer(args)