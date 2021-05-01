from torch import nn
from torch.nn import Sequential, Conv3d, InstanceNorm3d, LeakyReLU, ReLU, Linear, MaxPool3d, AvgPool3d, \
    AdaptiveAvgPool3d, Dropout
import torch
import torch.nn.functional as F
from itertools import combinations
from copy import deepcopy
import numpy as np


class ClassNet(nn.Module):
    def __init__(self, in_channels=128, base_channels=128, num_classes=5, ):
        super(ClassNet, self).__init__()
        self.fc1 = Linear(in_channels, base_channels)
        self.fc2 = Linear(base_channels, base_channels)
        self.fc3 = Linear(base_channels, num_classes)
        self.relu = ReLU()
        self.dropout = Dropout(inplace=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class cls_block(nn.Module):
    def __init__(self, in_channels=128, base_channels=128, num_classes=5, ):
        super(cls_block, self).__init__()
        self.fc1 = Linear(in_channels, base_channels)
        self.fc2 = Linear(base_channels, base_channels)
        self.fc3 = Linear(base_channels, num_classes)
        self.dropout = Dropout(inplace=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ClassNet_three(nn.Module):
    def __init__(self, in_channels=96, base_channels=128, num_classes=5, ):
        super(ClassNet_three, self).__init__()

        self.fc1 = nn.ModuleList([Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels)
                                  ])
        self.fc2 = nn.ModuleList([Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels)
                                  ])
        self.fc3 = nn.ModuleList([Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes)
                                  ])

        self.dropout = Dropout(inplace=True)

    def forward(self, x, idx):
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1[idx](x)))
        x = self.dropout(F.relu(self.fc2[idx](x)))
        x = self.fc3[idx](x)
        return x


class ClassNet_multi(nn.Module):
    def __init__(self, in_channels=32, base_channels=128, num_classes=5, num_phase=2):
        super(ClassNet_multi, self).__init__()

        in_channels = in_channels * num_phase
        combs = list(combinations(list(range(4)), num_phase))
        num_comb = len(combs)

        self.dropout = Dropout(inplace=True)

        layer1 = []
        layer2 = []
        layer3 = []

        for i in range(num_comb):
            layer1.append(Linear(in_channels, base_channels))
            layer2.append(Linear(base_channels, base_channels))
            layer3.append(Linear(base_channels, num_classes))

        self.fc1 = nn.ModuleList(layer1)
        self.fc2 = nn.ModuleList(layer2)
        self.fc3 = nn.ModuleList(layer3)

    def forward(self, x, idx):
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1[idx](x)))
        x = self.dropout(F.relu(self.fc2[idx](x)))
        x = self.fc3[idx](x)
        return x


class ClassNet_two(nn.Module):
    def __init__(self, in_channels=64, base_channels=128, num_classes=5, ):
        super(ClassNet_two, self).__init__()

        self.fc1 = nn.ModuleList([Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels),
                                  Linear(in_channels, base_channels)
                                  ])
        self.fc2 = nn.ModuleList([Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels),
                                  Linear(base_channels, base_channels)
                                  ])
        self.fc3 = nn.ModuleList([Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes),
                                  Linear(base_channels, num_classes)
                                  ])

        self.dropout = Dropout(inplace=True)

    def forward(self, x, idx):
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1[idx](x)))
        x = self.dropout(F.relu(self.fc2[idx](x)))
        x = self.fc3[idx](x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvBlock, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.norm = InstanceNorm3d(out_channels, affine=True)
        self.relu = LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvBlocks(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 num_blocks=2):
        super(ConvBlocks, self).__init__()
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride, padding))
        for i in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels, kernel_size, padding=padding))

        self.main = Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x


class SynNet(nn.Module):
    def __init__(self, in_channels=8, nCh=8):
        super(SynNet, self).__init__()
        self.down1 = ConvBlocks(in_channels, nCh)
        self.down2 = ConvBlocks(nCh, nCh * 2, stride=2)
        self.down3 = ConvBlocks(nCh * 2, nCh * 4, stride=2)
        self.down4 = ConvBlocks(nCh * 4, nCh * 8, stride=2)

        self.down5 = ConvBlocks(nCh * 8, nCh * 16, stride=2)

        self.up4 = ConvBlocks(nCh * 24, nCh * 8)
        self.up3 = ConvBlocks(nCh * 12, nCh * 4)
        self.up2 = ConvBlocks(nCh * 6, nCh * 2)
        self.up1 = ConvBlocks(nCh * 3, nCh)

        self.conv = Conv3d(nCh, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        up4 = self.up4(torch.cat((self.upsample(down5), down4), axis=1))
        up3 = self.up3(torch.cat((self.upsample(up4), down3), axis=1))
        up2 = self.up2(torch.cat((self.upsample(up3), down2), axis=1))
        up1 = self.up1(torch.cat((self.upsample(up2), down1), axis=1))

        out = self.conv(up1)
        out = torch.clamp(out, min=-1.522, max=2.463)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, nCh=8):
        super(Discriminator, self).__init__()
        self.down1 = ConvBlocks(in_channels, nCh, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                                num_blocks=1)
        self.down2 = ConvBlocks(nCh, nCh * 2, kernel_size=4, stride=(2, 2, 2), num_blocks=1)
        self.down3 = ConvBlocks(nCh * 2, nCh * 4, kernel_size=4, stride=(2, 2, 2), num_blocks=1)
        # self.down4 = ConvBlocks(nCh * 4, nCh * 8, kernel_size=4, stride=(2, 2, 2), num_blocks=1)
        # self.down5 = ConvBlocks(nCh * 8, nCh * 16, kernel_size=4, stride=(2, 2, 2), num_blocks=1)

        self.conv = Conv3d(nCh * 4, out_channels, 4, padding=0)
        # self.avgpool = AdaptiveAvgPool3d(1)
        # self.conv_cls = Conv3d(nCh*16, out_channels, 1)

    def forward(self, x, y=None):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x_dis = self.conv(x)
        # x = self.down4(x)
        # x = self.down5(x)
        # x = self.avgpool(x)
        # x_cls = self.conv_cls(x)

        if y is not None:
            return x_dis[:, y]
        else:
            return x_dis  # , x_cls.view(x_cls.size(0), -1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool

        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        self.num_classes = None  # number of channels in the output

        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonlin(nn.Module):

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=nn.Softmax, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, feat_only=False):

        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin(dim=1)
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.feat_only = feat_only

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x, y=None, use_seg=True, get_seg=False):
        skips = []
        features = []
        seg_outputs = []

        if y is not None:
            x = self.conv_blocks_context[0](x)
            mask = (y == 1).to(torch.float)
            # mask = (y[:, 1:2]).detach()
            area = torch.sum(mask.view(mask.size(0), -1), dim=1, keepdim=True)
            masked = x * mask
            masked = torch.sum(masked.view(masked.size(0), masked.size(1), -1), axis=2)
            masked = masked / area
            return masked.view(1, -1)

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if use_seg:
            mask = (seg_outputs[-1][:, 1:2]).detach()
            # mask = torch.mean(mask, dim=0, keepdim=True)
            area = torch.sum(mask.view(mask.size(0), -1), dim=1, keepdim=True)
            masked = skips[0] * mask
            masked = torch.sum(masked.view(masked.size(0), masked.size(1), -1), axis=2)
            masked = masked / area
            if get_seg:
                return masked.view(1, -1), seg_outputs[-1]
            else:
                return masked.view(1, -1)

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):

        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):

    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1., tumor_only=False, kidney_only=False):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.kidney_only = kidney_only
        self.tumor_only = tumor_only

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                if self.kidney_only:
                    dc = dc[:, 2]
                elif self.tumor_only:
                    dc = dc[:, 1]
                else:
                    dc = dc[:, 1:]

        # return dc

        dc = dc.mean()
        return 1. -dc