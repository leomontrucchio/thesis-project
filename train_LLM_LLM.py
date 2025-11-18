import argparse
import os
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange
import logging
import warnings
import torchvision.transforms.functional as F

from utils_visa.loader_VisA import get_data_loader
from utils_visa.general_utils import set_seeds

from models.teacher import LLMFeatureExtractor
from models.student import FeatureProjectionMLP


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)


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
    fe = LLMFeatureExtractor(layer1_idx=0, layer2_idx=1).to(device).eval()

    # Model instantiation.
    backward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device=device, dtype=torch.bfloat16)
    forward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(params = chain(backward_net.parameters(), forward_net.parameters()), lr = 1e-4)

    cos_sim = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for _ in trange(args.epochs_no, desc = f'Training students...'):

        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc = f'    Extracting features from class: {args.class_name}.'):
            backward_net.train(), forward_net.train()

            pil_img = F.to_pil_image(tensor_img)

            # 1. Feature extraction.
            with torch.no_grad():
                earlier_patch, later_patch = fe(pil_img)

            # 2. Nets prediction.
            predicted_later_patch = forward_net(earlier_patch)
            predicted_earlier_patch = backward_net(later_patch)

            # 3. Losses.
            loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()

            loss = loss_later + loss_earlier

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

    torch.save(backward_net.state_dict(), os.path.join(directory, 'backward_net_' + model_name + '.pth'))
    torch.save(forward_net.state_dict(), os.path.join(directory, 'forward_net_' + model_name + '.pth'))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training framework')

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

    parser.add_argument('--label', default = 'assistant_inspect_db_LLM_0_1', type = str, 
                        help = 'Label to identify the experiment.')

    args = parser.parse_args()
    train(args)