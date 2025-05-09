import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from models.base_model import BaseModel

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
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # self.args = args
        self.encoder = Densenet121(pretrained=pretrained)

        self.feature_dim = self.encoder.model.classifier.in_features

        self.fc = nn.Linear(self.feature_dim, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x, get_feature=False):
        fea = self.encoder(x)
        fea = F.normalize(self.fc(fea), dim=-1)
        out = self.out(fea)
        if get_feature:
            return out, fea
        return out

class ERMModel(BaseModel):
    def __init__(self, num_classes, *args, pretrained=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Network(num_classes, pretrained=pretrained)
        self.criterion = nn.BCELoss()
    
    def forward(self, images, sensitive_attribute_vals=None, get_feature=False):
        if get_feature:
            out, fea = self.model(images, get_feature=True)
            return torch.sigmoid(out), fea
        return torch.sigmoid(self.model(images))
        
if __name__ == '__main__':
    model = ERMModel.load_from_checkpoint('checkpoints/erm/refusion_model-epoch=05-val_loss=0.51.ckpt', num_classes=1, pretrained=True)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/erm/', filename='refusion_model-{epoch:02d}-{val_loss:.2f}', save_top_k=3, mode='min')
    trainer = Trainer(max_epochs=10, strategy='ddp_find_unused_parameters_true')
    trainer.test(model)