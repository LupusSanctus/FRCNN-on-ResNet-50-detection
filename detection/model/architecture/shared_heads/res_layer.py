import torch.nn as nn

from ..backbones import ResNet, make_res_layer
from ..cnn import constant_init, kaiming_init
from ...utils.fp16 import auto_fp16
from ...utils.registry_objects import SHARED_HEADS


@SHARED_HEADS.register_module
class ResLayer(nn.Module):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None):
        super(ResLayer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2 ** stage
        inplanes = 64 * 2 ** (stage - 1) * block.expansion

        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
