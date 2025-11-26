import argparse
import os
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange

from utils_loco.loader_loco import get_data_loader 
from utils_loco.general_utils import set_seeds

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float32 if args.students_blocks == 'Both ViT' else torch.bfloat16

    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project='L2BT_MLLM',
        name=model_name,
        # mode="disabled"
    )

    # Dataloaders.
    train_loader, _ = get_data_loader(
        "train", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )
    
    val_loader, _ = get_data_loader(
        "validation", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

    # Model Initialization.
    students = {}
    teacher = None

    if args.students_blocks == 'Both ViT':
        # --- ViT Bidirectional, Residual MLP ---
        
        # Teacher
        teacher = ViTFeatureExtractor(layers=[1, 5]).to(device).eval()
        
        # Students (Residual Architecture)
        students['backward_net'] = ResidualFeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, reduction_factor=1
        ).to(device)
        students['forward_net'] = ResidualFeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, reduction_factor=1
        ).to(device)

    elif args.students_blocks == 'Both LLM':
        # --- LLM Bidirectional, Standard MLP ---
        
        # Teacher
        teacher = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()
        
        # Students (Standard Architecture)
        students['backward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim
        ).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim
        ).to(device=device, dtype=dtype)

    else:
        raise ValueError("Invalid students_blocks option")

    # Optimizer
    optimizer = torch.optim.Adam(
        params=chain(students['backward_net'].parameters(), students['forward_net'].parameters()), 
        lr=1e-4
    )
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # --- Training Loop ---
    for epoch in trange(args.epochs_no, desc='Training students...'):

        for model in students.values():
            model.train()
            
        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc=f'    Epoch {epoch+1} [Train]'):
            
            # Prepare Input
            if args.students_blocks == 'Both ViT':
                input_data = tensor_img.to(device)
            else:
                input_data = pil_img

            # Feature extraction
            with torch.no_grad():
                earlier_patch, later_patch = teacher(input_data)

            # Nets prediction
            predicted_later_patch = students['forward_net'](earlier_patch)
            predicted_earlier_patch = students['backward_net'](later_patch)

            # Losses
            loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()
            loss = loss_later + loss_earlier

            # Logging & Optimization
            global_loss.append(loss.item())
            wandb.log({"train/batch_loss": loss.item()})

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is NaN/Inf. Exiting.")
                exit()

        epoch_train_loss = torch.tensor(global_loss).mean()
        wandb.log({"train/epoch_loss": epoch_train_loss.item()})


        # --- Validation ---
        for model in students.values():
            model.eval()
            
        global_val_loss = []

        with torch.no_grad():
            for pil_img, tensor_img in tqdm(val_loader, desc=f'    Epoch {epoch+1} [Val]'):
                
                # Prepare Input
                if args.students_blocks == 'Both ViT':
                    input_data = tensor_img.to(device)
                else:
                    input_data = pil_img

                earlier_patch, later_patch = teacher(input_data)

                predicted_later_patch = students['forward_net'](earlier_patch)
                predicted_earlier_patch = students['backward_net'](later_patch)

                loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
                loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()
                loss = loss_later + loss_earlier

                global_val_loss.append(loss.item())

        epoch_val_loss = torch.tensor(global_val_loss).mean()
        wandb.log({"val/epoch_loss": epoch_val_loss.item()})

        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # --- Model Saving ---
    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(directory, exist_ok=True)

    torch.save(students['forward_net'].state_dict(), os.path.join(directory, 'forward_net_' + model_name + '.pth'))
    torch.save(students['backward_net'].state_dict(), os.path.join(directory, 'backward_net_' + model_name + '.pth'))

    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Training Framework')

    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_loco', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = 'breakfast_box', type = str,
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')
    
    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    parser.add_argument('--students_blocks', type=str, default='Both ViT', choices=['Both ViT', 'Both LLM'],
                        help='Training scenario: where the 2 students extract features from')
    
    parser.add_argument('--label', default='unified_loco_exp', type=str,
                        help='Label to identify the experiment.')

    args = parser.parse_args()
    train(args)