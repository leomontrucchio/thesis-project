import argparse
import os
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange
import logging
import warnings


from utils_mvtec_loco.loader_loco import get_data_loader 
from utils_visa.general_utils import set_seeds

from models.teacher import ViTFeatureExtractor
from models.student import ResidualFeatureProjectionMLP


os.environ["TOKENIZERS_PARALLELISM"] = "false"
#logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
#logging.getLogger("transformers").setLevel(logging.ERROR)
#warnings.filterwarnings("ignore", category=FutureWarning)


def train(args):

    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'L2BT_MLLM',
        name = model_name,
        # mode = "disabled"
    )

    # Dataloaders.
    train_loader, _ = get_data_loader(
        "train", class_name = args.class_name,
        img_size = args.img_size, batch_size = args.batch_size,
        dataset_path = args.dataset_path)
    
    val_loader, _ = get_data_loader(
        "validation", class_name = args.class_name,
        img_size = args.img_size, batch_size = args.batch_size,
        dataset_path = args.dataset_path)

    # Feature extractor.
    fe = ViTFeatureExtractor(layers = [1,5]).to(device).eval()

    # Model instantiation.
    backward_net = ResidualFeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim, reduction_factor=1).to(device)
    forward_net = ResidualFeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim, reduction_factor=1).to(device)

    optimizer = torch.optim.Adam(params = chain(backward_net.parameters(), forward_net.parameters()), lr = 1e-4)

    cos_sim = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for epoch in trange(args.epochs_no, desc = f'Training students...'):

        # ------------ [Training Loop] ------------ #

        backward_net.train(), forward_net.train()
        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc = f'    Epoch {epoch+1} [Train]'):
            
            tensor_img = tensor_img.to(device)

            # 1. Feature extraction.
            with torch.no_grad():
                earlier_patch, later_patch = fe(tensor_img)

            # 2. Nets prediction.
            predicted_later_patch = forward_net(earlier_patch)
            predicted_earlier_patch = backward_net(later_patch)

            # 3. Losses.
            loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()
            loss = loss_later + loss_earlier

            # 4. Logging.
            global_loss.append(loss.item())

            wandb.log({"train/batch_loss" : loss.item()})

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                exit()


        epoch_train_loss = torch.Tensor(global_loss).mean()
        wandb.log({"train/epoch_loss" : epoch_train_loss})
        

        # ------------ [Validation Loop] ------------ #
        
        backward_net.eval(), forward_net.eval()
        global_val_loss = []

        with torch.no_grad():
            for pil_img, tensor_img in tqdm(val_loader, desc = f'    Epoch {epoch+1} [Val]'):
                
                tensor_img = tensor_img.to(device)

                earlier_patch, later_patch = fe(tensor_img)

                predicted_later_patch = forward_net(earlier_patch)
                predicted_earlier_patch = backward_net(later_patch)

                loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
                loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()
                loss = loss_later + loss_earlier

                global_val_loss.append(loss.item())

        epoch_val_loss = torch.Tensor(global_val_loss).mean()
        wandb.log({"val/epoch_loss" : epoch_val_loss})

        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # --- End of Training Loop ---

    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(forward_net.state_dict(), os.path.join(directory, 'forward_net_' + model_name + '.pth'))
    torch.save(backward_net.state_dict(), os.path.join(directory, 'backward_net_' + model_name + '.pth'))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training framework')

    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_loco', type = str, 
                        help = 'Where to save the model checkpoints.')

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
    train(args)