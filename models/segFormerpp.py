import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from mitDecode import MiTDecode


class SegFormerpp(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19, head: str = 'B0') -> None:
        super().__init__(backbone, num_classes)
        self.num_classes = num_classes
        self.head = head
        self.decode = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, 3)
        self.decode_head = MiTDecode(head)

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        print(y[0].shape)
        y = self.decode(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        y = self.decode_head(y)
        print(y[0].shape)
        y = self.decode(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


if __name__ == '__main__':
    model = SegFormerpp('MiT-B0')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)