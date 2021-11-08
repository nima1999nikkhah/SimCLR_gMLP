import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from icecream import ic
import os
from dataset import MyDataLoader
from tqdm import tqdm
import numpy as np
from gMLP import gMLPFeatures
import argparse


parser = argparse.ArgumentParser(description='Self Supervised Learning')
'''Data Arguments'''
parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py/',
                    help='ImageNet directory')
parser.add_argument('--resize_height', type=int, default=32,
                    help='Image resize height')
parser.add_argument('--resize_width', type=int, default=32,
                    help='Image resize width')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Dataloader workers')
'''Training Arguments'''
parser.add_argument('--batch_size', type=int, default=20,
                    help='Training batch size')
parser.add_argument('--lr_base_net', type=int, default=1e-4,
                    help='Learning rate for base network')
parser.add_argument('--lr_head_net', type=int, default=1e-1,
                    help='Learning rate for head (projection) network')
parser.add_argument('--wd', type=int, default=1e-6,
                    help='Learning weight decay')
parser.add_argument('--mlp_dim', type=int, default=128,
                    help='Dimension of FullyConnected middle Layer')
parser.add_argument('--proj_dim', type=int, default=64,
                    help='Dimension of FullyConnected Layer for projection')
'''Report Arguments'''
parser.add_argument('--report_iter', type=int, default=1,
                    help='Frequency of report')
parser.add_argument('--model_name', type=str, default='simCLR',
                    help='Name of the model (based on loss)')


args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImagePretraining(nn.Module):
    def __init__(self, middle_mlp_dim=512, proj_size=256):
        super(ImagePretraining, self).__init__()

        self.backbone = gMLPFeatures(
            survival_prob=0.99,
            image_size=32,
            patch_size=2,
            dim=128,
            depth=15,
            ff_mult=6)
        backbone_out_size = 128
        rep_size = proj_size

        self.rep_layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(backbone_out_size, middle_mlp_dim)),
            ('gn1', nn.GroupNorm(num_groups=1,
                                 num_channels=middle_mlp_dim)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(middle_mlp_dim, rep_size)),
            ('gn2', nn.GroupNorm(num_groups=1,
                                 num_channels=rep_size))
        ]))

        self.proj_layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(rep_size, middle_mlp_dim)),
            ('gn', nn.GroupNorm(num_groups=1,
                                num_channels=middle_mlp_dim)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(middle_mlp_dim, proj_size))
        ]))

    def contrastive_loss_simCLR(self, features1, features2, temperature=0.1):
        LARGE_NUM = 1e9

        feat1 = F.normalize(features1, p=2, dim=1)
        feat2 = F.normalize(features2, p=2, dim=1)

        batch_size = feat1.shape[0]

        y = torch.ones((batch_size)).to(device)

        labels = torch.arange(batch_size).to(device)
        masks = torch.diag(LARGE_NUM * y)

        logits_aa = torch.matmul(feat1, feat1.T) / temperature
        logits_aa = logits_aa - masks

        logits_bb = torch.matmul(feat2, feat2.T) / temperature
        logits_bb = logits_bb - masks

        logits_ab = torch.matmul(feat1, feat2.T) / temperature

        logits_ba = torch.matmul(feat2, feat1.T) / temperature

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        return logits_a, logits_b, labels

    def forward(self, x):
        features = self.backbone(x)
        rep = self.rep_layer(features)
        proj = self.proj_layer(rep)

        return rep, proj


def train(model, criterion, optimizer, scheduler, loader):
    model = model.train()
    batch_size = args.batch_size
    num_workers = args.num_workers
    report_iter = args.report_iter
    avg_loss = 0
    iteration = 0
    log_dir = 'runs/log-model_{}-batch_size_{}-mlp_dim_{}-proj_dim_{}-resize_{}-lr_base_{}-lr_head_{}'.format(
        args.model_name,
        args.batch_size,
        args.mlp_dim,
        args.proj_dim,
        args.resize_height,
        args.lr_base_net,
        args.lr_head_net)
    ic(log_dir)
    model_dir = 'checkpoint-model_{}-batch_size_{}-mlp_dim_{}-proj_dim_{}-resize_{}-lr_base_{}-lr_head_{}'.format(
        args.model_name,
        args.batch_size,
        args.mlp_dim,
        args.proj_dim,
        args.resize_height,
        args.lr_base_net,
        args.lr_head_net)
    ic(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    train_ds = loader.dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        train=True)
    for epoch in range(100):
        with tqdm(train_ds, unit="batch ({})".format(batch_size)) as t_iter:
            for sample in t_iter:

                iteration += 1
                imgs1, imgs2 = sample['imgs1'].to(
                    device), sample['imgs2'].to(device)
                optimizer.zero_grad()
                rep1, proj1 = model(imgs1)
                rep2, proj2 = model(imgs2)

                logits_a, logits_b, labels = model.contrastive_loss_simCLR(
                    proj1, proj2)
                simCLR_loss_1 = criterion(logits_a, labels)
                simCLR_loss_2 = criterion(logits_b, labels)
                loss = 0.5 * (simCLR_loss_1 + simCLR_loss_2)

                loss.backward()
                optimizer.step()

                avg_loss += loss
                if iteration % report_iter == 0:
                    avg_loss = avg_loss / report_iter
                    t_iter.set_postfix(avg_loss=avg_loss.detach().cpu().numpy())
                    writer.add_scalar(
                        'Loss/train',
                        avg_loss.detach().cpu().numpy(),
                        iteration)
                    avg_loss = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir,
                'model_epoch_{}.pth'.format(epoch)))

        if epoch >= 10:
            scheduler.step()


if __name__ == "__main__":

    loader = MyDataLoader(data_dir=args.data_dir,
                          resize_height=args.resize_height,
                          resize_width=args.resize_width)

    model = ImagePretraining(middle_mlp_dim=args.mlp_dim,
                             proj_size=args.proj_dim)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(),

                           lr=args.lr_base_net,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args.wd,
                           amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0, last_epoch=-1)
    train(model, criterion, optimizer, scheduler, loader)
