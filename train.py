import torch
import yaml

print(torch.__version__)
import torch.optim as optim
import os

from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from models.resnet_simclr import ResNetSimCLR
from utils import get_negative_mask, get_similarity_function, get_train_validation_data_loaders
from data_aug.data_transform import DataTransform, get_data_transform_opes

torch.manual_seed(0)
np.random.seed(0)

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

batch_size = config['batch_size']
out_dim = config['out_dim']
temperature = config['temperature']
use_cosine_similarity = config['use_cosine_similarity']

data_augment = get_data_transform_opes(s=config['s'], crop_size=96)

train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True, transform=DataTransform(data_augment))

train_loader, valid_loader = get_train_validation_data_loaders(train_dataset, config)

# model = Encoder(out_dim=out_dim)
model = ResNetSimCLR(base_model=config["base_convnet"], out_dim=out_dim)

train_gpu = torch.cuda.is_available()
print("Is gpu available:", train_gpu)

# moves the model parameters to gpu
if train_gpu:
    model.cuda()

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), 3e-4)

train_writer = SummaryWriter()

sim_func_dim1, sim_func_dim2 = get_similarity_function(use_cosine_similarity)

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(batch_size)


def step(xis, xjs):
    # get the representations and the projections
    ris, zis = model(xis)  # [N,C]
    train_writer.add_histogram("xi_repr", ris, global_step=n_iter)
    train_writer.add_histogram("xi_latent", zis, global_step=n_iter)

    # get the representations and the projections
    rjs, zjs = model(xjs)  # [N,C]
    train_writer.add_histogram("xj_repr", rjs, global_step=n_iter)
    train_writer.add_histogram("xj_latent", zjs, global_step=n_iter)

    # normalize projection feature vectors
    zis = F.normalize(zis, dim=1)
    zjs = F.normalize(zjs, dim=1)

    l_pos = sim_func_dim1(zis, zjs).view(batch_size, 1)
    l_pos /= temperature

    negatives = torch.cat([zjs, zis], dim=0)

    loss = 0
    for positives in [zis, zjs]:
        l_neg = sim_func_dim2(positives, negatives)

        labels = torch.zeros(batch_size, dtype=torch.long)
        if train_gpu:
            labels = labels.cuda()

        l_neg = l_neg[negative_mask].view(l_neg.shape[0], -1)
        l_neg /= temperature

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N,K+1]
        loss += criterion(logits, labels)

    loss = loss / (2 * batch_size)
    return loss


model_checkpoints_folder = os.path.join(train_writer.log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)

n_iter = 0
best_valid_loss = np.inf

for epoch_counter in range(config['epochs']):
    for (xis, xjs), _ in train_loader:
        optimizer.zero_grad()

        if train_gpu:
            xis = xis.cuda()
            xjs = xjs.cuda()

        loss = step(xis, xjs)

        train_writer.add_scalar('train_loss', loss, global_step=n_iter)
        loss.backward()
        optimizer.step()
        n_iter += 1

    if epoch_counter % config['eval_every_n_epochs'] == 0:

        # validation steps
        with torch.no_grad():
            model.eval()
            for (xis, xjs), _ in valid_loader:

                if train_gpu:
                    xis = xis.cuda()
                    xjs = xjs.cuda()

                loss = step(xis, xjs)

                if loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                train_writer.add_scalar('validation_loss', loss, global_step=epoch_counter)

        model.train()
