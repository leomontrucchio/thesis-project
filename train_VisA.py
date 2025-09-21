import argparse
import os
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange

from utils.loader_VisA import get_data_loader
from utils.general_utils import set_seeds

from models.teacher import FeatureExtractor
from models.students import FeatureProjectionMLP


def train(args):

    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'L2BT_MLLM',
        name = model_name,
        # mode = "disabled"
    )

    # Dataloader.
    train_loader, _ = get_data_loader(
        "train", class_name = args.class_name,
        img_size = args.img_size, batch_size = args.batch_size,
        dataset_path = args.dataset_path)

    # Feature extractor.
    vit_fe = ViTFeatureExtractor(layers = [7,11]).to(device, dtype=torch.bfloat16).eval()
    llm_fe = LLMFeatureExtractor().to(device).eval()

    # Model instantiation.
    vit_net = FeatureProjectionMLP(in_features=vit_fe.embed_dim, out_features=vit_fe.embed_dim, act_layer=torch.nn.GELU).to(device=device, dtype=torch.bfloat16)
    llm_net = FeatureProjectionMLP(in_features=llm_fe.embed_dim, out_features=llm_fe.embed_dim, act_layer=torch.nn.SiLU).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(params = chain(vit_net.parameters(), llm_net.parameters()), lr = 1e-4)

    cos_sim = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for _ in trange(args.epochs_no, desc = f'Training students...'):

        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc = f'    Extracting features from class: {args.class_name}.'):
            vit_net.train(), llm_net.train()

            tensor_img = tensor_img.to(device, dtype=torch.bfloat16)

            # 1. Feature extraction.
            with torch.no_grad():
                vit_1st_feats, vit_2nd_feats = vit_fe(tensor_img)
                llm_1st_feats, llm_2nd_feats = llm_fe(pil_img)

            # 2. Nets prediction.
            predicted_vit_feats = vit_net(vit_1st_feats)
            predicted_llm_feats = llm_net(llm_1st_feats)

            # 3. Losses.
            loss_vit = 1 - cos_sim(predicted_vit_feats, vit_2nd_feats).mean()
            loss_llm = 1 - cos_sim(predicted_llm_feats, llm_2nd_feats).mean()

            loss = loss_vit + loss_llm

            # 4. Logging.
            global_loss.append(loss.item())

            wandb.log({
                "train/batch_loss" : loss,
                })

            # 5. Optimization.
            if torch.isnan(loss) or torch.isinf(loss):
                exit()
            elif not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Global logging.
        wandb.log({
            "train/epoch_loss" : torch.Tensor(global_loss, device = 'cpu').mean(),
            })

    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(vit_net.state_dict(), os.path.join(directory, 'vit_net_' + model_name + '.pth'))
    torch.save(llm_net.state_dict(), os.path.join(directory, 'llm_net_' + model_name + '.pth'))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training framework')

    parser.add_argument('--dataset_path', default = './DATASETS/visa', type = str, 
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_visa', type = str, 
                        help = 'Where to save the model checkpoints.')

    parser.add_argument('--class_name', default = "candle", type = str,
                        help = 'Category name.')

    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')

    parser.add_argument('--img_size', default = 1036, type = int,
                        help = 'Square image resolution.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--label', default = 'final_model', type = str, 
                        help = 'Label to identify the experiment.')

    args = parser.parse_args()
    train(args)