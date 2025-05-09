import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from dataset.dataset_utils import age_to_class
from models.disentangled_model import Network as DisentangledNetwork
from models.base_model import BaseModel
from models.sa_model import SensitiveAttributeModel
 
class CascadeRescaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mean_std_dim=None, conv=('resnet', 4), dropout=0.5, norm_rescale=False):
        super(CascadeRescaleBlock, self).__init__()
        if mean_std_dim is None:
            mean_std_dim = in_channels
        if conv[0] == 'resnet':
            mid_channels = in_channels // conv[1]
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            # '2convs'
            mid_channels = (in_channels + out_channels) // 2
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.proj_to_rescale = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, mean_std_dim),
            nn.ReLU(),
        )

        self.proj_from_rescale = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mean_std_dim, in_channels),
            nn.ReLU(),
        )
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_rescale = norm_rescale

    def forward(self, x_in, mean, std):
        # project to rescale
        rescale_vec = self.proj_to_rescale(x_in)
        # normalize rescale_vec
        if self.norm_rescale:
            rescale_vec = F.normalize(rescale_vec, dim=-1)
        # rescale
        rescale_vec = rescale_vec * std + mean
        # project back
        rescale_vec = self.proj_from_rescale(rescale_vec)
        # do rescale on x_in
        rescale_vec = rescale_vec.unsqueeze(-1).unsqueeze(-1)
        x_in = x_in * rescale_vec
        # conv block
        x = self.conv_block(x_in)
        # residual connection
        # x = x + x_in
        return x

class RefusionCascadeRescale(BaseModel):
    def __init__(self, num_classes, 
                 num_cascade_blocks=2, dropout=0.5, mean_std_dim=None, conv=('resnet', 4), 
                 sa_encoder='simple', d_encoder='densenet', age_attr='AGE_AT_CXR',
                 *args, d_model_path=None, freeze_d_model=False, adv_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.disentangled_model = DisentangledNetwork(num_classes, encoder=d_encoder)

        if d_encoder == 'densenet':
            d_rep_dim = self.disentangled_model.encoder.model.classifier.in_features
        else:
            d_rep_dim = self.disentangled_model.out.in_features
        
        if mean_std_dim is None:
            mean_std_dim = d_rep_dim

        if d_model_path:
            self.disentangled_model.load_state_dict(torch.load(d_model_path)["state_dict"])
        else:
            raise ValueError("Disentangled model path is required for refusion model")
        
        # Only use the encoder part of the disentangled model TODO ablation on this
        self.disentangled_model = self.disentangled_model.encoder.model.features

        if freeze_d_model:
            for param in self.disentangled_model.parameters():
                param.requires_grad = False
            self.disentangled_model.eval()

        if sa_encoder == 'simple':
            self.sa_encoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(len(kwargs['sensitive_attributes']), mean_std_dim // 2),
                nn.ReLU(),
            )
        else:
            # 'multi-layer'
            self.sa_encoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(len(kwargs['sensitive_attributes']), mean_std_dim // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mean_std_dim // 8, mean_std_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mean_std_dim // 4, mean_std_dim // 2),
                nn.ReLU(),
            )
        self.mean_dropout = nn.Dropout(dropout)
        self.mean_decoder = nn.Linear(mean_std_dim // 2, mean_std_dim)
        self.std_dropout = nn.Dropout(dropout)
        self.std_decoder = nn.Linear(mean_std_dim // 2, mean_std_dim)

        # print('Conv is:', conv)
        if conv[0] == 'resnet':
            self.cascade_blocks = nn.ModuleList([CascadeRescaleBlock(d_rep_dim, d_rep_dim, mean_std_dim, conv, dropout) for _ in range(num_cascade_blocks)])
        else:
            # '2convs'
            conv[1] = list(conv[1])
            start_channels, end_channels = conv[1][0], conv[1][1]
            num_channels = [start_channels + (end_channels - start_channels) * i // num_cascade_blocks for i in range(num_cascade_blocks + 1)]
            self.cascade_blocks = []
            for i in range(num_cascade_blocks):
                self.cascade_blocks.append(CascadeRescaleBlock(num_channels[i], num_channels[i+1], mean_std_dim, conv, dropout))
            self.cascade_blocks = nn.ModuleList(self.cascade_blocks)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # adversarial sensitive attribute model
        if adv_alpha is not None:
            mid_channels = d_rep_dim // 8
            self.adv_sa_model = nn.Sequential(
                nn.Conv2d(d_rep_dim, mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, d_rep_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(d_rep_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(d_rep_dim, len(kwargs['sensitive_attributes'])),
                nn.Sigmoid(),
            )
            self.adv_criterion = nn.BCELoss()
            # self.automatic_optimization = False
        
        # classifier input size is the output size of the last conv block in cascade block
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.cascade_blocks[-1].conv_block[-3].out_channels, num_classes)
        self.criterion = nn.BCELoss()
        self.freeze_d_model = freeze_d_model
        self.num_cascade_blocks = num_cascade_blocks
        self.dropout = dropout
        self.adv_alpha = adv_alpha
        self.age_attr = age_attr

    def forward(self, x, sensitive_attribute, get_feature=False):
        # x: (b, c, h, w)
        # sensitive_attribute: (b, sa_dimension)
        # x: (b, d_rep_dim, 7, 7)
        if self.freeze_d_model:
            with torch.no_grad():
                x = self.disentangled_model(x)
        else:
            x = self.disentangled_model(x)
        disentangled_output = x.clone()
        # sa_feature: (b, 64)
        sa_feature = self.sa_encoder(sensitive_attribute)
        # mean: (b, d_rep_dim)
        # std: (b, d_rep_dim)
        sa_feature_for_mean = self.mean_dropout(sa_feature)
        sa_feature_for_std = self.std_dropout(sa_feature)
        mean = self.mean_decoder(sa_feature_for_mean)
        std = self.std_decoder(sa_feature_for_std)

        # cascade rescale
        for i in range(self.num_cascade_blocks):
            x = self.cascade_blocks[i](x, mean, std)

        x = self.pooling(x).squeeze()
        
        x = self.classifier_dropout(x)
        output = self.classifier(x)
        output = torch.sigmoid(output)
        if self.adv_alpha is not None:
            return output, disentangled_output
        elif get_feature:
            return output, x
        return output

    def training_step(self, batch, batch_idx):
        images, labels, sensitive_attribute_vals = batch
        
        # Forward pass through the main model
        outputs, disentangled_outputs = self(images, sensitive_attribute_vals)
        outputs = outputs.squeeze(1) if len(outputs.shape) > 1 else outputs
        loss = self.criterion(outputs, labels)
        self.log('train_loss_ce', loss)
        
        # Forward pass through the adversarial model
        sa_outputs = self.adv_sa_model(disentangled_outputs)
        sa_loss = self.adv_criterion(sa_outputs, sensitive_attribute_vals)
        self.log('train_loss_adv_sa', sa_loss.item())
        
        # Adjust the main model loss
        loss -= self.adv_alpha * sa_loss  # Encourage the main model to make sa_loss larger
        self.log('train_loss', loss)
        self.train_fairness_agg[0].append(outputs.detach())
        self.train_fairness_agg[1].append(labels.detach())
        self.train_fairness_agg[2].append(sensitive_attribute_vals.detach())
        
        return loss

    def validation_step(self, batch, batch_idx, log_prefix='val', fairness_agg_ls=None, print_calibration=False):
        if hasattr(self, 'adv_sa_model') and self.adv_alpha is not None:
            loss, disentangled_outputs = super().validation_step(batch, batch_idx, log_prefix, fairness_agg_ls)
            _, _, sensitive_attribute_vals = batch
            sa_outputs = self.adv_sa_model(disentangled_outputs)
            sensitive_attribute_vals[:, self.sensitive_attributes.index(self.age_attr)] *= 100
            sensitive_attribute_vals[:, self.sensitive_attributes.index(self.age_attr)] = age_to_class(sensitive_attribute_vals[:, self.sensitive_attributes.index(self.age_attr)])
            sa_loss = self.adv_criterion(sa_outputs, sensitive_attribute_vals)
            self.log(f'{log_prefix}_adv_sa_loss', sa_loss.item())
            loss -= self.adv_alpha * sa_loss
            return loss, disentangled_outputs
        else:
            loss = super().validation_step(batch, batch_idx, log_prefix, fairness_agg_ls)
            return loss
    
    def test_step(self, batch, batch_idx, log_prefix='test'):
        return self.validation_step(batch, batch_idx, log_prefix, fairness_agg_ls=self.test_fairness_agg, print_calibration=True)
    
    # def configure_optimizers(self):
    #     if hasattr(self, 'adv_sa_model') and self.adv_alpha is not None:
    #         optimizer_main = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #         optimizer_adv = torch.optim.Adam(self.adv_sa_model.parameters(), lr=1e-4)
    #         return [optimizer_main, optimizer_adv]
    #     else:
    #         return super().configure_optimizers()
