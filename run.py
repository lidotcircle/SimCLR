import argparse
import torch
import torch.backends.cudnn as cudnn
import os
import pathlib
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.cvt import CvT
from models.resnet_simclr import ResNetSimCLR
from models.simple_resnet import ResNet18
from simclr import SimCLR
from tdlogger import TdLogger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names = model_names + ['cvt']

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--dataroot', type=str, required=True,
                    help='dataset root dir',)
parser.add_argument('--test-dataroot', type=str, required=True,
                    help='test dataset root dir')
parser.add_argument('--name', type=str, required=True,
                    help='model name')
parser.add_argument('--test-interval', default=1, type=int,
                    help='epoch interval for evaluating a batch of test sample')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_count', default=0, type=int, help='start epoch')
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'eval'], help='evaluate model and produce heatmap(CAM-like)')
parser.add_argument('--aug_transforms', type=str, default='crop_flip_color_gray_blur', help='augmentation operations')
parser.add_argument('--eval-dataroot', type=str, default='', help='evaluation dataroot')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=1, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('--logger_endpoint', type=str , default="http://192.168.44.43:5445", help='logger endpoint')
parser.add_argument('--logger_prefix',   type=str,  default="", help='logger group prefix')
parser.add_argument('--disable_logger',  action='store_true', help='ignore logging request')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    train_dataset = ContrastiveLearningDataset(args.dataroot).get_dataset(args.n_views, aug_transforms=args.aug_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_dataset = ContrastiveLearningDataset(args.test_dataroot).get_dataset(args.n_views, aug_transforms=args.aug_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    logger = TdLogger(args.logger_endpoint, "Loss", 1, ("admin", "123456"), group_prefix=args.logger_prefix + "SimCLR", disabled=args.disable_logger)

    model = None
    if args.arch == 'resnet18':
        model = ResNet18(num_outputs=args.out_dim)
    elif args.arch == 'cvt':
        model = CvT(num_classes=args.out_dim, s3_depth=6)
    else:
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)


    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", args.name)
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(logger=logger, checkpoint_dir=checkpoint_dir, model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.phase == 'train':
            simclr.train(train_loader, test_loader, args.test_interval)
        elif args.phase == 'test':
            simclr.test(test_loader)
        else:
            eval_dataset = ContrastiveLearningDataset(args.eval_dataroot).get_dataset(1, identical=True)
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            simclr.eval(eval_loader)


if __name__ == "__main__":
    main()
