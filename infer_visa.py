import argparse
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils_visa.loader_VisA import get_data_loader
from utils_visa.general_utils import set_seeds, siglip_denormalize
from utils_visa.metrics_utils import calculate_au_pro

# Import Models
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
        
        students['backward_net'] = ResidualFeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device)
        students['forward_net'] = ResidualFeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device)

        # Load Checkpoints
        fwd_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        
        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'ViT-LLM':
        # --- ViT & LLM forward ---
        teachers['vit_fe'] = ViTFeatureExtractor(layers=[7, 11]).to(device, dtype=dtype).eval()
        teachers['llm_fe'] = LLMFeatureExtractor().to(device).eval()
        
        students['vit_net'] = FeatureProjectionMLP(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim).to(device=device, dtype=dtype)
        students['llm_net'] = FeatureProjectionMLP(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim).to(device=device, dtype=dtype)

        # Load Checkpoints
        vit_path = os.path.join(check_dir, f'vit_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_path = os.path.join(check_dir, f'llm_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['vit_net'].load_state_dict(torch.load(vit_path, map_location=device, weights_only=False))
        students['llm_net'].load_state_dict(torch.load(llm_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Both LLM':
        # --- LLM Bidirectional ---
        teachers['fe'] = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()
        
        students['backward_net'] = FeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim).to(device=device, dtype=dtype)

        # Load Checkpoints
        fwd_path = os.path.join(check_dir, f'forward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Full':
        # --- ViT Bidirectional + LLM Bidirectional ---
        teachers['vit_fe'] = ViTFeatureExtractor(layers=[1, 5]).to(device, dtype=dtype).eval()
        teachers['llm_fe'] = LLMFeatureExtractor(layer1_idx=0, layer2_idx=2).to(device).eval()

        # ViT Students (Residual)
        students['vit_fw'] = ResidualFeatureProjectionMLP(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, reduction_factor=1).to(device=device, dtype=dtype)
        students['vit_bw'] = ResidualFeatureProjectionMLP(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, reduction_factor=1).to(device=device, dtype=dtype)
        
        # LLM Students (Standard)
        students['llm_fw'] = FeatureProjectionMLP(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim).to(device=device, dtype=dtype)
        students['llm_bw'] = FeatureProjectionMLP(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim).to(device=device, dtype=dtype)

        # Load Checkpoints
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

    # Initialize Lists
    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    inference_time = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # --- Inference Loop ---
    for pil_img, tensor_img, gt, label, rgb_path in tqdm(test_loader, desc=f'Inferencing {args.class_name}'):
        
        defect_class_str = rgb_path[0].split('/')[-3]
        image_name_str = rgb_path[0].split('/')[-1]

        with torch.no_grad():
            tensor_img = tensor_img.to(device, dtype=dtype)
            
            start_event.record()

            # Feature Extraction & Prediction & Map Calculation
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

            elif args.students_blocks == 'ViT-LLM':
                # Extraction
                vit_1st, vit_2nd = teachers['vit_fe'](tensor_img)
                llm_1st, llm_2nd = teachers['llm_fe'](pil_img)
                # Prediction
                pred_vit = students['vit_net'](vit_1st)
                pred_llm = students['llm_net'](llm_1st)
                # Map Calc
                vit_map = (torch.nn.functional.normalize(pred_vit, dim=-1) - torch.nn.functional.normalize(vit_2nd, dim=-1)).pow(2).sum(-1).sqrt()
                llm_map = (torch.nn.functional.normalize(pred_llm, dim=-1) - torch.nn.functional.normalize(llm_2nd, dim=-1)).pow(2).sum(-1).sqrt()
                
                vit_reshaped = vit_map.reshape(1, 1, 27, 27)
                llm_reshaped = llm_map.reshape(1, 1, 14, 14)
                llm_resized = torch.nn.functional.interpolate(llm_reshaped, size=(27, 27), mode='bilinear', align_corners=False)
                
                combined_anomaly_map = vit_reshaped * llm_resized

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

            # Upsample to original resolution.
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=[max_hw, max_hw], mode='bilinear')

            # Approximated Gaussian blur.
            for _ in range(5):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_u, weight=weight_u)

            combined_anomaly_map = combined_anomaly_map.reshape(max_hw, max_hw)
            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, gt.shape[-2:])

            end_event.record()
            torch.cuda.synchronize()
            inference_time.append(start_event.elapsed_time(end_event))

            # Accumulation
            map_np = combined_anomaly_map.to(torch.float32).cpu().detach().numpy()
            gt_np = gt.squeeze().cpu().detach().numpy()

            gts.append(gt_np)
            predictions.append(map_np)
            image_labels.append(label)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())

            # Score calculation
            K = int((map_np.shape[0] * map_np.shape[1]) * 0.001)
            global_score = torch.topk(combined_anomaly_map.flatten(), k=K)[0].mean().to(torch.float32).cpu().detach().numpy()
            
            image_preds.append(global_score)
            pixel_preds.extend(map_np.flatten())

            # Qualitative Plotting
            if args.produce_qualitatives:
                save_path = f'{args.qualitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'
                os.makedirs(save_path, exist_ok=True)

                _, axs = plt.subplots(1, 3, figsize=(7, 3))

                img = siglip_denormalize(tensor_img)
                img = torch.nn.functional.interpolate(img, size=[max_hw, max_hw], mode='bilinear')
                img = torchvision.transforms.functional.center_crop(img, gt.shape[-2:])
                
                axs[0].imshow(img.squeeze().permute(1, 2, 0).to(torch.float32).cpu().detach().numpy())
                axs[0].set_title('Input Image')
                axs[1].imshow(gt_np, cmap='gray')
                axs[1].set_title('Ground-truth')
                axs[2].imshow(map_np, cmap=plt.cm.jet)
                axs[2].set_title('Anomaly Map')

                for ax in axs.flat:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, image_name_str), dpi=256)
                if args.visualize_plot:
                    plt.show()
                plt.close()

    # Calculate AD&S metrics.
    au_pros_q4, _, weights = calculate_au_pro(gts, predictions, weighted=False)
    q1, q2, q3 = np.quantile(weights, 0.25), np.quantile(weights, 0.5), np.quantile(weights, 0.75)
    
    au_pros_q3, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q3))
    au_pros_q2, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q2))
    au_pros_q1, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q1))

    pixel_rocauc = roc_auc_score(np.array(pixel_labels), np.array(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    result_file_name = f'{args.quantitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    
    # Print AD&S metrics.
    print(f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs')
    print("P-AUROC  |  I-AUROC")
    print(f'  {pixel_rocauc:.3f}   |   {image_rocauc:.3f}', end = '\n')

    print(" QUARTILE | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |")
    print(f'    Q4    |   {au_pros_q4[0]:.3f}   |   {au_pros_q4[1]:.3f}   |   {au_pros_q4[2]:.3f}  |   {au_pros_q4[3]:.3f}  |', end = '\n')
    print(f'    Q3    |   {au_pros_q3[0]:.3f}   |   {au_pros_q3[1]:.3f}   |   {au_pros_q3[2]:.3f}  |   {au_pros_q3[3]:.3f}  |', end = '\n')
    print(f'    Q2    |   {au_pros_q2[0]:.3f}   |   {au_pros_q2[1]:.3f}   |   {au_pros_q2[2]:.3f}  |   {au_pros_q2[3]:.3f}  |', end = '\n')
    print(f'    Q1    |   {au_pros_q1[0]:.3f}   |   {au_pros_q1[1]:.3f}   |   {au_pros_q1[2]:.3f}  |   {au_pros_q1[3]:.3f}  |', end = '\n')


    os.makedirs(args.quantitative_folder, exist_ok=True)

    # Writing MD
    with open(result_file_name, "w") as f:
        f.write(f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs\n')
        f.write('Cumulative\nAUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC\n')
        f.write(f'{au_pros_q4[0]:.3f} & {au_pros_q4[1]:.3f} & {au_pros_q4[2]:.3f} & {au_pros_q4[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}\n')
        f.write(f'{au_pros_q3[0]:.3f} & {au_pros_q3[1]:.3f} & {au_pros_q3[2]:.3f} & {au_pros_q3[3]:.3f}\n')
        f.write(f'{au_pros_q2[0]:.3f} & {au_pros_q2[1]:.3f} & {au_pros_q2[2]:.3f} & {au_pros_q2[3]:.3f}\n')
        f.write(f'{au_pros_q1[0]:.3f} & {au_pros_q1[1]:.3f} & {au_pros_q1[2]:.3f} & {au_pros_q1[3]:.3f}\n')

    # Saving CSV
    results = {
        'class_name': [args.class_name], 'inference_time': [np.mean(inference_time)], 
        'i_auroc': [image_rocauc], 'p_auroc': [pixel_rocauc],
        'aupro_30_q4': [au_pros_q4[0]], 'aupro_10_q4': [au_pros_q4[1]], 'aupro_05_q4': [au_pros_q4[2]], 'aupro_01_q4': [au_pros_q4[3]],
        'aupro_30_q3': [au_pros_q3[0]], 'aupro_10_q3': [au_pros_q3[1]], 'aupro_05_q3': [au_pros_q3[2]], 'aupro_01_q3': [au_pros_q3[3]],
        'aupro_30_q2': [au_pros_q2[0]], 'aupro_10_q2': [au_pros_q2[1]], 'aupro_05_q2': [au_pros_q2[2]], 'aupro_01_q2': [au_pros_q2[3]],
        'aupro_30_q1': [au_pros_q1[0]], 'aupro_10_q1': [au_pros_q1[1]], 'aupro_05_q1': [au_pros_q1[2]], 'aupro_01_q1': [au_pros_q1[3]],
    }
    pd.DataFrame(results).to_csv(result_file_name.replace('md', 'csv'), index=False, sep='&')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference Framework')

    parser.add_argument('--checkpoint_folder', default = './checkpoints/checkpoints_visa', type = str,
                        help = 'Path to the folder containing students checkpoints.')
    
    parser.add_argument('--qualitative_folder', default = './results/visa/qualitatives_visa', type = str,
                        help = 'Path to the folder in which to save the qualitatives.')
    
    parser.add_argument('--quantitative_folder', default = './results/visa/quantitatives_visa', type = str,
                        help = 'Path to the folder in which to save the quantitatives.')
    
    parser.add_argument('--visualize_plot', default = False, action = 'store_true',
                        help = 'Whether to show plot or not.')
    
    parser.add_argument('--produce_qualitatives', default = True, action = 'store_true',
                        help = 'Whether to produce qualitatives or not.')
    
    parser.add_argument('--dataset_path', default = './datasets/visa', type = str,
                        help = 'Dataset path.')
    
    parser.add_argument('--class_name', default = "candle", type = str,
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')
    
    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--students_blocks', type=str, default='Full', choices=['Both ViT', 'ViT-LLM', 'Both LLM', 'Full'],
                        help='Inference scenario.')
    
    parser.add_argument('--label', default='ground_db_LLM_0_2', type=str,
                        help='Experiment label. For experiment involving just ViT students, use the same name of vit_label.')
    
    parser.add_argument('--vit_label', default='l2bt_deepseek_res', type=str, 
                        help='Label of experiment involving just ViT students.')

    args = parser.parse_args()

    infer(args)