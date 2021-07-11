import os
import torch
import argparse
from networks import get_model
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from utils.loss_functions import calc_loss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

seed.seed_everything(123)
parser = argparse.ArgumentParser('train')
parser.add_argument('--train_data_path', type=str, default='/data/kits/train')
parser.add_argument('--test_data_path', type=str, default='/data/kits/test')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model', type=str, default='raunet')
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-2)


class SegPL(BasePLModel):
    def __init__(self, params):
        super(SegPL, self).__init__()
        self.save_hyperparameters(params)
        self.net = get_model(self.hparams.model, channels=2)

    def forward(self, x):
        output, _, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)  # Dice_loss Used

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)

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
    model = SegPL(args)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_%s_{epoch}' % (args.dataset, args.task, args.model),
        save_last=True,
        save_top_k=-1,
    )

    logger = TensorBoardLogger('log', name='%s_%s_%s' % (args.dataset, args.task, args.model))
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epochs, gpus=[8], callbacks=checkpoint_callback, logger=logger)
    trainer.fit(model)


def test():
    args = parser.parse_args()
    model = SegPL.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt'))
    trainer = Trainer(gpus=[8])
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
