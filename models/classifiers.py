import torch
import torch.nn as nn
import torchvision.models as models

class RobustClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10):
        super(RobustClassifier, self).__init__()
        self.defense_layer = None
        self.model_name = model_name
        
        if model_name == 'resnet18':
            model = models.resnet18(num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Sequential(nn.Flatten(), model.fc)
        elif model_name == 'resnet50':
            model = models.resnet50(num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Sequential(nn.Flatten(), model.fc)
        elif model_name == 'vgg16':
            model = models.vgg16(num_classes=num_classes)
            self.features = model.features
            self.classifier = nn.Sequential(
                model.avgpool,
                nn.Flatten(),
                model.classifier
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def set_defense(self, defense):
        self.defense_layer = defense
        
    def get_latent_features(self, x):
        f = self.features(x)
        return torch.flatten(f, 1)
        
    def forward(self, x):
        f = self.features(x)
        f_flat = torch.flatten(f, 1)
        
        if self.defense_layer is not None:
            f_flat_reconstructed = self.defense_layer.reconstruct(f_flat)
            f = f_flat_reconstructed.view(f.size())
            
        return self.classifier(f)
