import os

import torch
import torch.nn.functional as F
import GPUtil
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import accuracy, save_checkpoint
from tdlogger import TdLogger

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, logger: TdLogger, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def send_gpuinfo(self):
        gpus = GPUtil.getGPUs()
        gpu_loadinfo = {}
        gpu_meminfo = {}
        for i in range(0, len(gpus)):
            gpu = gpus[i]
            gpu_loadinfo["GPU" + str(i) + " Load"] = gpu.load
            gpu_meminfo["GPU" + str(i) + " MemUsed"] = gpu.memoryUsed
            gpu_meminfo["GPU" + str(i) + " MemFree"] = gpu.memoryFree
            gpu_meminfo["GPU" + str(i) + " MemTotal"] = gpu.memoryTotal
        self.logger.send(gpu_loadinfo, "gpuload_info")
        self.logger.send(gpu_meminfo,  "gpumem_info")

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.size(0) // 2) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader: DataLoader, test_loader: DataLoader, test_interval: int):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        n_iter = 0
        self.logger.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        self.logger.info(f"Training with gpu: {self.args.disable_cuda}.")

        def eval_test():
            un = enumerate(test_loader)
            _, imgs = next(un)
            imgs = torch.cat(imgs, dim=0)
            imgs = imgs.to(self.args.device)
            features = self.model(imgs)
            logits, labels = self.info_nce_loss(features)
            loss = self.criterion(logits, labels)
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            stats = {}
            stats['loss'] = loss.item()
            stats['top1'] = top1.mean().item()
            stats['top5'] = top5.mean().item()
            self.logger.send(stats, "validation", True)

        for epoch_counter in range(self.args.epochs):
            for images, images2 in tqdm(train_loader):
                images = torch.cat([images, images2], dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    stats = {}
                    stats['loss'] = loss.item()
                    stats['top1'] = top1.mean().item()
                    stats['top5'] = top5.mean().item()
                    self.logger.send(stats)
                    self.send_gpuinfo()

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            self.logger.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            if epoch_counter % test_interval == 0:
                with torch.no_grad():
                    eval_test()

        self.logger.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        self.logger.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
