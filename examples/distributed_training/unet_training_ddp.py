# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to execute distributed training based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed training:

- Execute `torch.distributed.launch` to create processes on every node for every GPU.
  It receives parameters as below:
  `--nproc_per_node=NUM_GPUS_PER_NODE`
  `--nnodes=NUM_NODES`
  `--node_rank=INDEX_CURRENT_NODE`
  `--master_addr="192.168.1.1"`
  `--master_port=1234`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
  Alternatively, we can also use `torch.multiprocessing.spawn` to start program, but it that case, need to handle
  all the above parameters and compute `rank` manually, then set to `init_process_group`, etc.
  `torch.distributed.launch` is even more efficient than `torch.multiprocessing.spawn` during training.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and must set `init_method="env://"` if use `torch.distributed.launch`.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Wrap Dataset with `DistributedSampler`, and disable the `shuffle` in DataLoader.
  Instead, shuffle data by `train_sampler.set_epoch(epoch)` before every epoch.

Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="192.168.1.1" --master_port=1234
           unet_training_ddp.py -d DIR_OF_TESTDATA

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import argparse
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)


def train(args):
    # disable logging for processes execpt 0 on every node
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    elif not os.path.exists(args.dir):
        # create 40 random image, mask paris for training
        print(f"generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        # set random seed to generate same random data for every node
        np.random.seed(seed=0)
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"seg{i:d}.nii.gz"))

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # create a training data sampler
    train_sampler = DistributedSampler(train_ds)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{args.local_rank}")
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # start a typical PyTorch training
    epoch_loss_values = list()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    print(f"train completed, epoch losses: {epoch_loss_values}")
    if dist.get_rank() == 0:
        # all processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes,
        # therefore, saving it in one process is sufficient
        torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


import dllogger as logger
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend
import time


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class LoggingCallback:
    def __init__(self, logger, batch_size, num_gpus=1, warmup_steps=50, mode='train'):
        self._logger = logger
        self._batch_size = batch_size
        self._num_gpus = num_gpus
        self._warmup_steps = warmup_steps
        self._step = 0
        self._timestamps = []
        self._mode = mode

    def on_batch_start(self):
        self._step += 1
        if self._step >= self._warmup_steps:
            torch.cuda.synchronize()
            self._timestamps.append(time.time())

    def on_fit_end(self):
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)])
        stats = process_performance_stats(np.array(deltas),
                                          self._batch_size * self._num_gpus,
                                          self._mode)
        if is_main_process():
            self._logger.log(step=(), data={metric: float(value) for (metric, value) in stats})
            self._logger.flush()

    # def on_test_batch_start(self, trainer, pl_module):
    #     self.on_batch_start(trainer, pl_module)
    #
    # def on_test_end(self, trainer, pl_module):
    #     self.on_fit_end(trainer)
    #
    # def on_validation_end(self, trainer, pl_module):
    #     if is_main_process():
    #         self._logger.log(step=(pl_module.current_epoch,), data={"val_loss": pl_module._val_loss,
    #                                                                 **pl_module._val_dice})
    #         self._logger.flush()


def process_performance_stats(timestamps, batch_size, mode):
    timestamps_ms = 1000 * timestamps
    latency_ms = timestamps_ms.mean()
    std = timestamps_ms.std()
    n = np.sqrt(len(timestamps_ms))
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()

    stats = [("throughput_{}".format(mode), str(throughput_imgps)),
             ('latency_{}:'.format(mode), str(latency_ms))]
    for ci, lvl in zip(["90%:", "95%:", "99%:"],
                       [1.645, 1.960, 2.576]):
        stats.append(("Latency_{} ".format(mode) + ci, str(latency_ms + lvl * std / n)))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    train(args=args)


# usage example(refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        unet_training_ddp.py -d DIR_OF_TESTDATA

if __name__ == "__main__":
    main()
