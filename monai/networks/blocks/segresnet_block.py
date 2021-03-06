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

import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Norm


def get_norm_layer(spatial_dims: int, in_channels: int, norm_name: str, num_groups: int = 8):
    if norm_name not in ["batch", "instance", "group"]:
        raise ValueError(f"Unsupported normalization mode: {norm_name}")
    else:
        if norm_name == "group":
            norm = Norm[norm_name](num_groups=num_groups, num_channels=in_channels)
        else:
            norm = Norm[norm_name, spatial_dims](in_channels)
        if norm.bias is not None:
            nn.init.zeros_(norm.bias)
        if norm.weight is not None:
            nn.init.ones_(norm.weight)
        return norm


def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):

    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        bias=bias,
        conv_only=True,
    )


def get_upsample_layer(spatial_dims: int, in_channels: int, upsample_mode: str = "trilinear", scale_factor: int = 2):
    up_module: nn.Module
    if upsample_mode == "transpose":
        up_module = UpSample(
            spatial_dims,
            in_channels,
            scale_factor=scale_factor,
            with_conv=True,
        )
    else:
        upsample_mode = "bilinear" if spatial_dims == 2 else "trilinear"
        up_module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode, align_corners=False)
    return up_module


class ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        norm_name: str = "group",
        num_groups: int = 8,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            stride: convolution stride. Defaults to 1.
            bias: whether to have a bias term in convolution layer. Defaults to ``True``.
            norm_name: feature normalization type, this module only supports group norm,
                batch norm and instance norm. Defaults to ``group``.
            num_groups: number of groups to separate the channels into, in this module,
                in_channels should be divisible by num_groups. Defaults to 8.
        """

        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size should be an odd number."
        assert in_channels % num_groups == 0, "in_channels should be divisible by num_groups."

        self.norm1 = get_norm_layer(spatial_dims, in_channels, norm_name, num_groups=num_groups)
        self.norm2 = get_norm_layer(spatial_dims, in_channels, norm_name, num_groups=num_groups)
        self.relu = Act[Act.RELU](inplace=True)
        self.conv1 = get_conv_layer(spatial_dims, in_channels, in_channels)
        self.conv2 = get_conv_layer(spatial_dims, in_channels, in_channels)

    def forward(self, x):

        identity = x

        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += identity

        return x
