from __future__ import absolute_import, division, print_function

import os
import sys
import time
from collections import namedtuple
from statistics import mean

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from got10k.trackers import Tracker
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from . import ops
from .backbones import AlexNetV1
from .datasets import Pair
from .heads import SiamFC
from .losses import BalancedLoss
from .transforms import SiamFCTransforms

__all__ = ['TrackerSiamFCLongTerm']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFCLongTerm(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFCLongTerm, self).__init__('SiamFC', True)

        self.kernel = None
        self.avg_color = None
        self.x_size = None
        self.z_size = None
        self.scale_factors = None
        self.hanning_window = None
        self.upscale_size = None
        self.target_size = None
        self.center = None

        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum
        )

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num
        )
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

        self.failure_threshold = 3.5
        self.redetection_threshold = None
        self.redetection_samples = 20

        self.sampling_method = "gauss"  # random/gauss
        self.gauss_cov = 4500
        self.target_visible = True
        self.frame_index = 0

        self.target_correlations = []
        self.initial_target_correlation = None

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,  # 32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def random_samples(self, method, img_size):
        if method == "random":
            x_positions = np.random.randint(10, img_size[0], self.redetection_samples).astype("float")
            y_positions = np.random.randint(10, img_size[1], self.redetection_samples).astype("float")
            return np.array([[x, y] for x, y in zip(x_positions, y_positions)])
        elif method == "gauss":
            return np.random.multivariate_normal(self.center, np.array([[self.gauss_cov, 0], [0, self.gauss_cov]]),
                                                 self.redetection_samples)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def init(self, img, box_):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box_ = np.array([
            box_[1] - 1 + (box_[3] - 1) / 2,
            box_[0] - 1 + (box_[2] - 1) / 2,
            box_[3], box_[2]], dtype=np.float32)
        self.center, self.target_size = box_[:2], box_[2:]

        # create Hanning window
        self.upscale_size = self.cfg.response_up * self.cfg.response_sz
        self.hanning_window = np.outer(np.hanning(self.upscale_size), np.hanning(self.upscale_size))
        self.hanning_window /= self.hanning_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(-(self.cfg.scale_num // 2), self.cfg.scale_num // 2,
                                                                self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_size)
        self.z_size = np.sqrt(np.prod(self.target_size + context))
        self.x_size = self.z_size * self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img,
            self.center,
            self.z_size,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color
        )

        # exemplar features
        z = torch.from_numpy(z).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        self.frame_index += 1
        start_time = time.time()

        # Target is visible
        if self.target_visible:
            # search images
            x = []
            for f in self.scale_factors:
                x.append(ops.crop_and_resize(
                    img,
                    self.center,
                    self.x_size * f,
                    out_size=self.cfg.instance_sz,
                    border_value=self.avg_color
                ))

        # Target is not visible -> re-detection
        else:
            # Random positions around previous seen position of target
            positions = self.random_samples(self.sampling_method, (img.shape[0], img.shape[1]))

            x = []
            for position in positions:
                x.append(ops.crop_and_resize(
                    img,
                    position,
                    self.x_size,
                    out_size=self.cfg.instance_sz,
                    border_value=self.avg_color
                ))

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x) \
            .to(self.device) \
            .permute(0, 3, 1, 2).float()

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # up-sample responses and penalize scale changes
        responses = np.stack(
            [cv2.resize(u, (self.upscale_size, self.upscale_size), interpolation=cv2.INTER_CUBIC) for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        max_response = max(0, response.max())
        # print(max_resp)

        # This happens only the first time
        if not self.initial_target_correlation:
            self.initial_target_correlation = max_response

        # Target correlations will be empty only the first time
        if not self.target_correlations or self.target_visible:
            self.target_correlations.append(max_response)
            self.redetection_threshold = mean(self.target_correlations) - 0.2

        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hanning_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # Locate target center
        disp_in_response = np.array(loc) - (self.upscale_size - 1) / 2
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up

        if self.target_visible:
            disp_in_image = disp_in_instance * self.x_size * self.scale_factors[scale_id] / self.cfg.instance_sz
            self.center += disp_in_image
            scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
        else:
            scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr

        self.target_size *= scale
        self.z_size *= scale
        self.x_size *= scale

        # Target is visible if max response is higher than threshold
        if not self.target_visible and max_response > self.redetection_threshold:
            self.target_visible = True
        elif self.target_visible and max_response < self.failure_threshold:
            self.target_visible = False

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_size[1] - 1) / 2,
            self.center[0] + 1 - (self.target_size[0] - 1) / 2,
            self.target_size[1], self.target_size[0]])

        if not self.target_visible:
            max_response = 0
        else:
            max_response = max_response / self.initial_target_correlation

        
        return box, max_response, time.time() - start_time

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
