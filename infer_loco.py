import argparse
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
import subprocess
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from utils_loco.loader_loco import get_data_loader
from utils_loco.general_utils import set_seeds, extract_metrics, siglip_denormalize

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import json
import subprocess
import pandas as pd

def run_evaluation_pipeline(args):
    eval_script_path = os.path.join("mvtec_loco_ad_evaluation", "evaluate_experiment.py")
    quantitative_dir = getattr(args, 'quantitative_folder', './results/loco/quantitatives_loco')
    os.makedirs(quantitative_dir, exist_ok=True)
    
    print(f"\nRunning evaluation for {args.class_name}...")
    
    # Run evaluate_experiment.py
    cmd = [
        "python3", eval_script_path,
        "--object_name", args.class_name,
        "--dataset_base_dir", args.dataset_path,
        "--anomaly_maps_dir", args.anomaly_maps_dir,
        "--output_dir", quantitative_dir,
        "--num_parallel_workers", "10", 
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        return

    # Parse Results
    metrics_path = os.path.join(quantitative_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"metrics.json not found at {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract for all categories
    glob_auc, glob_30, glob_10, glob_05, glob_01 = extract_metrics('global', metrics)
    struct_auc, struct_30, struct_10, struct_05, struct_01 = extract_metrics('structural_anomalies', metrics)
    logic_auc, logic_30, logic_10, logic_05, logic_01 = extract_metrics('logical_anomalies', metrics)

    # Print
    print(f"\nResults for {args.class_name}:")
    header = f"{'Type':<12} | {'I-AUROC':<8} | {'sPRO@30%':<8} | {'sPRO@10%':<8} | {'sPRO@5%':<8} | {'sPRO@1%':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Global':<12} | {glob_auc:.3f}    | {glob_30:.3f}    | {glob_10:.3f}    | {glob_05:.3f}   | {glob_01:.3f}")
    print(f"{'Structural':<12} | {struct_auc:.3f}    | {struct_30:.3f}    | {struct_10:.3f}    | {struct_05:.3f}   | {struct_01:.3f}")
    print(f"{'Logical':<12} | {logic_auc:.3f}    | {logic_30:.3f}    | {logic_10:.3f}    | {logic_05:.3f}   | {logic_01:.3f}")
    print("-" * len(header))

    # Write Markdown Report
    result_file_name = f'{quantitative_dir}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    
    with open(result_file_name, "w") as f:
        f.write(f'Metrics for class {args.class_name}\n')
        f.write(f'Config: {args.epochs_no} epochs, {args.batch_size} batch size\n\n')
        
        f.write('| Type       | sPRO@30% | sPRO@10% | sPRO@5%  | sPRO@1%  | I-AUROC  |\n')
        f.write('| ---------- | -------- | -------- | -------- | -------- | -------- |\n')
        f.write(f'| Global     | {glob_30:.3f} | {glob_10:.3f} | {glob_05:.3f} | {glob_01:.3f} | {glob_auc:.3f} |\n')
        f.write(f'| Structural | {struct_30:.3f} | {struct_10:.3f} | {struct_05:.3f} | {struct_01:.3f} | {struct_auc:.3f} |\n')
        f.write(f'| Logical    | {logic_30:.3f} | {logic_10:.3f} | {logic_05:.3f} | {logic_01:.3f} | {logic_auc:.3f} |\n')

    # Write CSV Report
    results = {
        'class_name': [args.class_name],
        # Global
        'global_i_auroc': [glob_auc],
        'global_spro_30': [glob_30], 'global_spro_10': [glob_10], 'global_spro_05': [glob_05], 'global_spro_01': [glob_01],
        # Structural
        'struct_i_auroc': [struct_auc],
        'struct_spro_30': [struct_30], 'struct_spro_10': [struct_10], 'struct_spro_05': [struct_05], 'struct_spro_01': [struct_01],
        # Logical
        'logic_i_auroc': [logic_auc],
        'logic_spro_30': [logic_30], 'logic_spro_10': [logic_10], 'logic_spro_05': [logic_05], 'logic_spro_01': [logic_01],
    }
    
    pd.DataFrame(results).to_csv(result_file_name.replace('md', 'csv'), index=False, sep=',')
    


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

            if args.produce_qualitatives:
                save_qual_path = f'{args.qualitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{parent_dir}'
                os.makedirs(save_qual_path, exist_ok=True)

                # Load Ground Truth
                if parent_dir == 'good':
                    gt_np = np.zeros((original_h, original_w))
                else:
                    gt_folder_path = os.path.join(args.dataset_path, args.class_name, 'ground_truth', parent_dir, file_name_no_ext)
                    gt_files = glob.glob(os.path.join(gt_folder_path, "*.png"))
                    
                    
                    # Combine all channels (Logical OR)
                    gt_combined = None
                    for f in gt_files:
                        mask = np.array(Image.open(f).convert('L'))
                        mask = (mask > 0).astype(np.float32)
                        if gt_combined is None:
                            gt_combined = mask
                        else:
                            gt_combined = np.maximum(gt_combined, mask)
                    gt_np = gt_combined

                # Prepare Plot
                _, axs = plt.subplots(1, 3, figsize=(7, 3))

                img_vis = siglip_denormalize(tensor_img)
                img_vis = torch.nn.functional.interpolate(img_vis, size=[max_hw, max_hw], mode='bilinear')
                img_vis = torchvision.transforms.functional.center_crop(img_vis, [original_h, original_w])
                
                axs[0].imshow(img_vis.squeeze().permute(1, 2, 0).to(torch.float32).cpu().detach().numpy())
                axs[0].set_title('Input Image')
                axs[1].imshow(gt_np, cmap='gray')
                axs[1].set_title('Ground-truth')
                axs[2].imshow(map_numpy, cmap=plt.cm.jet)
                axs[2].set_title('Anomaly Map')

                for ax in axs.flat:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_qual_path, file_name_no_ext), dpi=256)
                plt.close()

    print("Inference complete. Maps are stored.")

    run_evaluation_pipeline(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference Framework')

    parser.add_argument('--checkpoint_folder', default = './checkpoints/checkpoints_loco', type = str,
                        help = 'Path to the folder containing students checkpoints.')
    
    parser.add_argument('--anomaly_maps_dir', default='./results/loco/anomaly_maps', type=str,
                        help='Base directory where anomaly maps will be saved.')
    
    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str,
                        help = 'Dataset path.')
    
    parser.add_argument('--quantitative_folder', default='./results/loco/quantitatives_loco', type=str,
                        help='Path to the folder in which to save the quantitatives.')
    
    parser.add_argument('--qualitative_folder', default='./results/loco/qualitatives_loco', type=str,
                        help='Path to save qualitatives.')
    
    parser.add_argument('--produce_qualitatives', default=True, action='store_true',
                        help='Whether to produce qualitatives or not.')
    
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
    
    parser.add_argument('--label', default='assistant_inspect_db_LLM_0_1', type=str,
                        help='Experiment label. For experiment involving just ViT students, use the same name of vit_label.')
    
    parser.add_argument('--vit_label', default='l2bt_deepseek_res_new', type=str,
                        help='Label of experiment involving just ViT students.')

    args = parser.parse_args()
    infer(args)