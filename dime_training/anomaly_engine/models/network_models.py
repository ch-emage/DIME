import timm
import torchvision.models as models
import torchvision

_BACKBONES = {
    "alexnet": "models.alexnet(weights='IMAGENET1K_V1')",
    "resnet50": "models.resnet50(weights='IMAGENET1K_V2')",
    "resnet101": "models.resnet101(weights='IMAGENET1K_V2')",
    "resnext101": "models.resnext101_32x8d(weights='IMAGENET1K_V2')",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "vgg11": "models.vgg11(weights='IMAGENET1K_V1')",
    "vgg19": "models.vgg19(weights='IMAGENET1K_V1')",
    "vgg19_bn": "models.vgg19_bn(weights='IMAGENET1K_V1')",
    "wideresnet50": "models.wide_resnet50_2(weights='IMAGENET1K_V2')",
    "wide_resnet50_2": 'torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V2")',
    "wideresnet101": "models.wide_resnet101_2(weights='IMAGENET1K_V2')",
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "efficientnet_b0": 'timm.create_model("efficientnet_b0", pretrained=True)',  # Added
    "efficientnet_b1": 'timm.create_model("efficientnet_b1", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "mobilenetv3_large": 'timm.create_model("mobilenetv3_large_100", pretrained=True)',  # Added
}

def load(name):
    """Loads a pre-trained network model."""
    return eval(_BACKBONES[name])