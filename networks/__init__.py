from .deeplabv3_plus import DeepLabV3Plus
from .ENet import ENet
from .ERFNet import ERFNet
from .ESPNet import ESPNet
from .mobilenetv2 import MobileNetV2
from .NestedUNet import NestedUNet
from .RAUNet import RAUNet
from .resnet18 import Resnet18
from .UNet import U_Net
from .PspNet.pspnet import PSPNet


def get_model(model_name: str, channels: int):
    assert model_name.lower() in ['deeplabv3+', 'enet', 'erfnet', 'espnet', 'mobilenetv2',
                             'unet++', 'raunet', 'resnet18', 'unet', 'pspnet']
    if model_name.lower() == 'deeplabv3+':
        model = DeepLabV3Plus(num_class=channels)
    elif model_name.lower() == 'unet':
        model = U_Net(in_ch=1, out_ch=channels)
    elif model_name.lower() == 'resnet':
        model = Resnet18(num_classes=channels)
    elif model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'pspnet':
        model = PSPNet(num_classes=2)
    elif model_name.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=channels)
    elif model_name.lower() == 'espnet':
        model = ESPNet(classes=channels)
    elif model_name.lower() == 'erfnet':
        model = ERFNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(nclass=channels)
    elif model_name.lower() == 'unet++':
        model = NestedUNet(in_ch=1, out_ch=channels)
    return model