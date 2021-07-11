import torch
from pytorch_lightning.core import LightningModule


class BasePLModel(LightningModule):
    def __init__(self):
        super(BasePLModel, self).__init__()
        self.metric = {}
        self.num_class = 2

    def training_epoch_end(self, outputs):
        train_loss_mean = 0
        for output in outputs:
            train_loss_mean += output['loss']

        train_loss_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        self.log('train_loss', train_loss_mean)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def measure(self, batch, output):
        ct, mask, name = batch

        output = torch.softmax(output, dim=1)[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        # threshold value
        output = (output > 0.4).float()

        # record values concerned with dice score
        for ib in range(len(ct)):
            pre = torch.sum(output[ib], dim=(1, 2))
            gt = torch.sum(mask[ib], dim=(1, 2))
            inter = torch.sum(torch.mul(output[ib], mask[ib]), dim=(1, 2))
            if name[ib] not in self.metric.keys():
                self.metric[name[ib]] = torch.stack((pre, gt, inter), dim=0)
            else:
                self.metric[name[ib]] += torch.stack((pre, gt, inter), dim=0)

    def test_epoch_end(self, outputs):
        # calculate dice score
        num_class = self.num_class - 1
        scores = torch.zeros((num_class, 3))
        nums = torch.zeros((num_class, 1))
        for k, v in self.metric.items():
            dice = (2. * v[2] + 1.0) / (v[0] + v[1] + 1.0)
            voe = (2. * (v[0] - v[2])) / (v[0] + v[1] + 1e-7)
            rvd = v[0] / (v[1] + 1e-7) - 1.

            for i in range(num_class):
                # the dice is nonsensical when gt is 0
                if v[1][i].item() != 0:
                    nums[i] += 1
                    scores[i][0] += dice[i].item()
                    scores[i][1] += voe[i].item()
                    scores[i][2] += rvd[i].item()

        scores = scores / nums

        for i in range(num_class):
            # the dice is nonsensical when gt is 0
            self.log('dice_class{}'.format(i), scores[i][0].item())
            self.log('voe_class{}'.format(i), scores[i][1].item())
            self.log('rvd_class{}'.format(i), scores[i][2].item())

            print('dice_class{}: {}'.format(i, scores[i][0].item()))
            print('voe_class{}: {}'.format(i, scores[i][1].item()))
            print('rvd_class{}: {}'.format(i, scores[i][2].item()))

        self.metric = {}
