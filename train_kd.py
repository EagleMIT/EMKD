import os
import torch
import argparse
from train import SegPL
from networks import get_model
from utils.loss_functions import *
from torch.utils.data import DataLoader
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


seed.seed_everything(123)
parser = argparse.ArgumentParser('train_kd')
parser.add_argument('--train_data_path', type=str, default='/data/kits/train')
parser.add_argument('--test_data_path', type=str, default='/data/kits/test')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--tckpt', type=str, default='/data/checkpoints/checkpoint_kits_tumor_enet_epoch=18.ckpt', help='teacher model checkpoint path')
parser.add_argument('--smodel', type=str, default='enet')
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-2)

# KD loss para
alpha = 0.1
beta1 = 0.9
beta2 = 0.9


class KDPL(BasePLModel):
    def __init__(self, params):
        super(KDPL, self).__init__()
        self.save_hyperparameters(params)

        # load and freeze teacher net
        self.t_net = SegPL.load_from_checkpoint(checkpoint_path=self.hparams.tckpt)
        self.t_net.freeze()

        # student net
        self.net = get_model(self.hparams.smodel, channels=2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)

        loss_seg = calc_loss(output, mask)

        loss_pmd = prediction_map_distillation(output, t_out)
        loss_imd = importance_maps_distillation(low, t_low) + importance_maps_distillation(high, t_high)
        loss_rad = region_affinity_distillation(low, t_low, mask) + region_affinity_distillation(high, t_high, mask)

        loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd + beta2 * loss_rad

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output, low, high = self.net(ct)

        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.train_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=32, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.test_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            train=False
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]


def main():
    args = parser.parse_args()
    model = KDPL(args)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_kd_%s_{epoch}' % (args.dataset, args.task, args.smodel),
    )

    logger = TensorBoardLogger('log', name='%s_%s_kd_%s' % (args.dataset, args.task, args.smodel))
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epochs, gpus=[8], callbacks=checkpoint_callback, logger=logger)
    trainer.fit(model)


def test():
    args = parser.parse_args()
    model = KDPL.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt'))
    trainer = Trainer(gpus=args.gpu)
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()