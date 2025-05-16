import torch.nn as nn
import timm

class EfficientNetBirdCLEF(nn.Module):
    """BirdCLEF model using a timm EfficientNet backbone."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            drop_rate=0.2, # Consider making these configurable
            drop_path_rate=0.2 # Consider making these configurable
        )

        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out, config.num_classes)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits

def get_en_model(config_obj):
    """Helper function to instantiate the EfficientNetBirdCLEF model."""
    required_attrs = ['model_name', 'pretrained', 'in_channels', 'num_classes']
    if 'efficientnet' not in config_obj.model_name:
        raise ValueError(f"Model name {config_obj.model_name} is not an EfficientNet model.")
    for attr in required_attrs:
        if not hasattr(config_obj, attr):
            raise AttributeError(f"Config object is missing required attribute for EN model: {attr}")
            
    model = EfficientNetBirdCLEF(config_obj)
    return model
