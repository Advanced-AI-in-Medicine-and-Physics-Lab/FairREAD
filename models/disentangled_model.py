import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lightning import LightningModule, Trainer

from models.base_model import BaseModel

class ViTB16(nn.Module):
    def __init__(self, pretrained):
        super(ViTB16, self).__init__()
        self.model = models.vit_b_16(
            weights="DEFAULT" if pretrained else None
        )

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x
    
class Densenet121(nn.Module):
    def __init__(self, pretrained):
        super(Densenet121, self).__init__()
        self.model = models.densenet121(
           weights="DEFAULT" if pretrained else None
        )

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x


class Network(nn.Module):
    def __init__(self, num_classes, pretrained=True, encoder='densenet'):
        super().__init__()
        if encoder == 'densenet':
            self.encoder = Densenet121(pretrained=pretrained)
            self.feature_dim = self.encoder.model.classifier.in_features
        else:
            self.encoder = ViTB16(pretrained=pretrained)
            self.feature_dim = self.encoder.model.heads.head.in_features

        self.fc = nn.Linear(self.feature_dim, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x, get_feature=False):
        fea = self.encoder(x)
        fea = F.normalize(self.fc(fea), dim=-1)
        out = self.out(fea)
        if get_feature:
            return out, fea
        return out
    
    def get_feature(self, x):
        fea = self.encoder(x)
        return fea

    def get_projected_feature(self, x):
        fea = self.encoder(x)
        fea = F.normalize(self.fc(fea), dim=-1)
        return fea
    
class DisentangledModel(BaseModel):
    def __init__(self, num_classes, *args, d_checkpoint_path=None, pretrained=True, d_encoder='densenet', **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Network(num_classes, pretrained=pretrained, encoder=d_encoder)
        if d_checkpoint_path:
            print(f'Loading disentangled model from {d_checkpoint_path}')
            self.model.load_state_dict(torch.load(d_checkpoint_path)['state_dict'])
        self.criterion = nn.BCELoss()


    def forward(self, x, _, get_feature=False):
        if get_feature:
            out, fea = self.model(x, get_feature=True)
            return torch.sigmoid(out), fea
        return torch.sigmoid(self.model(x))

