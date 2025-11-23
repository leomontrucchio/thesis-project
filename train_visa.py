import argparse
import os
import torch
import wandb
import logging
import warnings
from itertools import chain
from tqdm import tqdm, trange
import torchvision.transforms.functional as F

from utils_visa.loader_VisA import get_data_loader
from utils_visa.general_utils import set_seeds

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=FutureWarning)


def train(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.float32 if args.students_blocks == 'Both ViT' else torch.bfloat16

    # Construct run name
    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project='L2BT_MLLM',
        name=model_name,
        # mode="disabled"
    )

    # Dataloader.
    train_loader, _ = get_data_loader(
        "train", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

    # Model Initialization
    students = {}
    teachers = {}
    
    optimizer_params = []

    if args.students_blocks == 'Both ViT':
        # --- ViT Bidirectional, Residual MLP ---
        
        # Teacher
        teachers['fe'] = ViTFeatureExtractor(layers=[1, 5]).to(device).eval()
        
        # Students
        students['backward_net'] = ResidualFeatureProjectionMLP(
            in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim, reduction_factor=1
        ).to(device)
        students['forward_net'] = ResidualFeatureProjectionMLP(
            in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim, reduction_factor=1
        ).to(device)

    elif args.students_blocks == 'ViT-LLM':
        # --- ViT & LLM forward, Standard MLP ---
        
        # Teachers
        teachers['vit_fe'] = ViTFeatureExtractor(layers=[7, 11]).to(device, dtype=dtype).eval()
        teachers['llm_fe'] = LLMFeatureExtractor().to(device).eval() # LLM usually handles its own dtype or runs fp16/bf16 internally
        
        # Students
        students['vit_net'] = FeatureProjectionMLP(
            in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, 
            act_layer=torch.nn.GELU
        ).to(device=device, dtype=dtype)
        
        students['llm_net'] = FeatureProjectionMLP(
            in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim, 
            act_layer=torch.nn.SiLU
        ).to(device=device, dtype=dtype)

    elif args.students_blocks == 'Both LLM':
        # --- LLM Bidirectional, Standard MLP ---
        
        # Teacher
        teachers['fe'] = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()
        
        # Students
        students['backward_net'] = FeatureProjectionMLP(
            in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim
        ).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(
            in_features=teachers['fe'].embed_dim, out_features=teachers['fe'].embed_dim
        ).to(device=device, dtype=dtype)

    else:
        raise ValueError("Select a valid students_blocks option")

    for name, model in students.items():
        optimizer_params.append(model.parameters())
    
    optimizer = torch.optim.Adam(params=chain(*optimizer_params), lr=1e-4)
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # --- Training Loop ---
    for epoch in trange(args.epochs_no, desc='Training students...'):
        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc=f'    Epoch {epoch+1} [Train]'):
            
            for model in students.values():
                model.train()

            tensor_img = tensor_img.to(device, dtype=dtype)

            # Feature extraction.
            with torch.no_grad():
                if args.students_blocks == 'Both ViT':
                    earlier_patch, later_patch = teachers['fe'](tensor_img)
                elif args.students_blocks == 'ViT-LLM':
                    vit_1st, vit_2nd = teachers['vit_fe'](tensor_img)
                    llm_1st, llm_2nd = teachers['llm_fe'](pil_img)
                elif args.students_blocks == 'Both LLM':
                    earlier_patch, later_patch = teachers['fe'](pil_img)

            # Nets prediction and losses.
            if args.students_blocks in ['Both ViT', 'Both LLM']:
                # Bidirectional logic (Forward + Backward)
                pred_later = students['forward_net'](earlier_patch)
                pred_earlier = students['backward_net'](later_patch)
                
                loss_later = 1 - cos_sim(pred_later, later_patch).mean()
                loss_earlier = 1 - cos_sim(pred_earlier, earlier_patch).mean()
                
                loss = loss_later + loss_earlier
                
            elif args.students_blocks == 'ViT-LLM':
                # Parallel logic (ViT stream + LLM stream)
                pred_vit = students['vit_net'](vit_1st)
                pred_llm = students['llm_net'](llm_1st)
                
                loss_vit = 1 - cos_sim(pred_vit, vit_2nd).mean()
                loss_llm = 1 - cos_sim(pred_llm, llm_2nd).mean()
                
                loss = loss_vit + loss_llm

            # Logging.
            global_loss.append(loss.item())
            wandb.log({"train/batch_loss": loss})

            # Optimization
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is NaN or Inf. Exiting.")
                exit()

        # Global logging
        wandb.log({"train/epoch_loss": torch.tensor(global_loss, device = 'cpu').mean()})

    # Model Saving
    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(directory, exist_ok=True)

    for name, model in students.items():
        save_name = f'{name}_{model_name}.pth'
        torch.save(model.state_dict(), os.path.join(directory, save_name))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Training Framework')

    parser.add_argument('--dataset_path', default = './datasets/visa', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_visa', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = "candle", type = str,
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')
    
    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    parser.add_argument('--students_blocks', type=str, default='Both ViT', choices=['Both ViT', 'ViT-LLM', 'Both LLM'],
                        help='Where the 2 students extract features from')
    
    parser.add_argument('--label', default = 'assistant_from_deepseek_MLLM', type = str, 
                        help = 'Label to identify the experiment.')

    args = parser.parse_args()
    train(args)