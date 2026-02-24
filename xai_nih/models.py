import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels = x.size()
        y = self.squeeze(x.unsqueeze(2)).squeeze(2)
        y = self.excitation(y)
        return x * y.expand_as(x)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        
    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention
class MinimalEnsembleModel(nn.Module):
    """Matches the actual checkpoint architecture - 15 output classes"""
    def __init__(self, num_classes=15):  # Changed to 15!
        super().__init__()
        self.num_classes = num_classes
        
        self.vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=15)
        self.swin_base = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=15) 
        self.convnext_base = timm.create_model('convnext_base', pretrained=True, num_classes=15)
        
        # Exact structure from checkpoint: 45 -> 256 -> 128 -> 15
        self.ensemble_layer = nn.Sequential(
            nn.Linear(45, 256),      # layer 0
            nn.ReLU(),               # layer 1
            nn.Dropout(0.3),         # layer 2
            nn.Linear(256, 128),     # layer 3
            nn.ReLU(),               # layer 4
            nn.Dropout(0.3),         # layer 5
            nn.Linear(128, 15)       # layer 6 - outputs 15 classes!
        )
        
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Each backbone outputs 15 classes
        vit_out = self.vit_base(x)      # [B, 15]
        swin_out = self.swin_base(x)    # [B, 15]
        convnext_out = self.convnext_base(x)  # [B, 15]
        
        # Concatenate all outputs
        combined = torch.cat([vit_out, swin_out, convnext_out], dim=1)  # [B, 45]
        
        # Ensemble prediction
        ensemble_output = self.ensemble_layer(combined)  # [B, 14]
        
        # Weighted average (take first 14 classes if outputs are 15)
        weights = F.softmax(self.attention_weights, dim=0)
        weighted_output = (
            vit_out[:, :self.num_classes] * weights[0] + 
            swin_out[:, :self.num_classes] * weights[1] + 
            convnext_out[:, :self.num_classes] * weights[2]
        )
        
        final_output = 0.6 * ensemble_output + 0.4 * weighted_output
        
        return {
            'ensemble': final_output,
            'vit_base': vit_out,
            'swin_base': swin_out,
            'convnext_base': convnext_out,
            'attention_weights': weights,
        }

class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x.unsqueeze(2)).squeeze(2))
        max_out = self.fc(self.max_pool(x.unsqueeze(2)).squeeze(2))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class MultiScaleFeatureFusion(nn.Module):
    """Advanced feature fusion with multi-scale processing"""
    def __init__(self, vit_dim, swin_dim, convnext_dim, output_dim=1024):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.swin_proj = nn.Sequential(
            nn.Linear(swin_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.convnext_proj = nn.Sequential(
            nn.Linear(convnext_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.vit_channel_att = ChannelAttention(output_dim)
        self.swin_channel_att = ChannelAttention(output_dim)
        self.convnext_channel_att = ChannelAttention(output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 3),
            nn.Sigmoid()
        )
        
    def forward(self, vit_feat, swin_feat, convnext_feat):
        vit_proj = self.vit_proj(vit_feat)
        swin_proj = self.swin_proj(swin_feat)
        convnext_proj = self.convnext_proj(convnext_feat)
        
        vit_att = self.vit_channel_att(vit_proj)
        swin_att = self.swin_channel_att(swin_proj)
        convnext_att = self.convnext_channel_att(convnext_proj)
        
        stacked = torch.stack([vit_att, swin_att, convnext_att], dim=1)
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        
        vit_cross = attended[:, 0, :]
        swin_cross = attended[:, 1, :]
        convnext_cross = attended[:, 2, :]
        
        concat_feat = torch.cat([vit_cross, swin_cross, convnext_cross], dim=1)
        gate_weights = self.gate(concat_feat)
        gated_feat = concat_feat * gate_weights
        
        return gated_feat


# OLD MODEL - This matches your checkpoint!
class TransformerEnsembleModel(nn.Module):
    """Original ensemble model - matches the checkpoint architecture"""
    def __init__(self, num_classes=14):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbones with OLD naming
        self.vit_base = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.swin_base = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        self.convnext_base = timm.create_model('convnext_base', pretrained=False, num_classes=0)
        
        vit_features = self.vit_base.num_features
        swin_features = self.swin_base.num_features
        convnext_features = self.convnext_base.num_features
        
        # Individual heads
        self.vit_head = nn.Sequential(
            nn.Linear(vit_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )
        
        self.swin_head = nn.Sequential(
            nn.Linear(swin_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )
        
        self.convnext_head = nn.Sequential(
            nn.Linear(convnext_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )
        
        total_features = vit_features + swin_features + convnext_features
        
        # Ensemble fusion layer - matches checkpoint!
        self.ensemble_layer = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.se_block = SEBlock(512, reduction=16)
        self.classifier = nn.Linear(512, num_classes)
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        vit_features = self.vit_base(x)
        swin_features = self.swin_base(x)
        convnext_features = self.convnext_base(x)
        
        vit_out = self.vit_head(vit_features)
        swin_out = self.swin_head(swin_features)
        convnext_out = self.convnext_head(convnext_features)
        
        combined_features = torch.cat([vit_features, swin_features, convnext_features], dim=1)
        fused_features = self.ensemble_layer(combined_features)
        fused_features = self.se_block(fused_features)
        ensemble_output = self.classifier(fused_features)
        
        weights = F.softmax(self.attention_weights, dim=0)
        weighted_output = (
            vit_out * weights[0] + 
            swin_out * weights[1] + 
            convnext_out * weights[2]
        )
        
        final_output = 0.6 * ensemble_output + 0.4 * weighted_output
        
        return {
            'ensemble': final_output,
            'vit_base': vit_out,
            'swin_base': swin_out,
            'convnext_base': convnext_out,
            'attention_weights': weights,
            'features': fused_features
        }


# NEW MODEL - Improved version
class ImprovedTransformerEnsembleModel(nn.Module):
    """Improved ensemble with multi-scale fusion and deep supervision"""
    def __init__(self, num_classes=Config.NUM_CLASSES, use_aux_loss=True):
        super(ImprovedTransformerEnsembleModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_aux_loss = use_aux_loss
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        
        vit_features = self.vit.num_features
        swin_features = self.swin.num_features
        convnext_features = self.convnext.num_features
        
        self.feature_fusion = MultiScaleFeatureFusion(
            vit_features, swin_features, convnext_features, output_dim=1024
        )
        
        self.ensemble_head = nn.Sequential(
            nn.Linear(1024 * 3, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE * 0.5),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE * 0.5),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE * 0.3),
            
            nn.Linear(512, num_classes)
        )
        
        if self.use_aux_loss:
            self.vit_aux = self._make_aux_head(vit_features, num_classes)
            self.swin_aux = self._make_aux_head(swin_features, num_classes)
            self.convnext_aux = self._make_aux_head(convnext_features, num_classes)
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def _make_aux_head(self, in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE * 0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE * 0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_features=False):
        vit_features = self.vit(x)
        swin_features = self.swin(x)
        convnext_features = self.convnext(x)
        
        fused_features = self.feature_fusion(vit_features, swin_features, convnext_features)
        ensemble_logits = self.ensemble_head(fused_features)
        ensemble_logits_scaled = ensemble_logits / self.temperature
        
        output_dict = {
            'logits': ensemble_logits_scaled,
            'ensemble': ensemble_logits_scaled,
        }
        
        if self.use_aux_loss and self.training:
            output_dict['vit_aux'] = self.vit_aux(vit_features)
            output_dict['swin_aux'] = self.swin_aux(swin_features)
            output_dict['convnext_aux'] = self.convnext_aux(convnext_features)
        
        if return_features:
            output_dict['features'] = fused_features
            output_dict['vit_features'] = vit_features
            output_dict['swin_features'] = swin_features
            output_dict['convnext_features'] = convnext_features
        
        return output_dict


class MCDropoutModel(nn.Module):
    def __init__(self, base_model, dropout_rate=Config.DROPOUT_RATE):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mc_samples=Config.MC_SAMPLES):
        if self.training:
            return self.base_model(x)
        else:
            predictions = []
            for _ in range(mc_samples):
                self.train()
                with torch.no_grad():
                    pred = self.base_model(x)
                    if isinstance(pred, dict):
                        pred = pred['ensemble']
                predictions.append(F.softmax(pred, dim=1))
            
            self.eval()
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.var(dim=0).mean(dim=1)
            
            return mean_pred, uncertainty

class SingleModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=Config.NUM_CLASSES, pretrained=True):
        super(SingleModel, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class VMambaModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(VMambaModel, self).__init__()
        try:
            self.model = timm.create_model('vmamba_base', pretrained=pretrained, num_classes=num_classes)
        except:
            self.model = timm.create_model('vssm_base_v2', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class DenseNet169Model(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(DenseNet169Model, self).__init__()
        
        self.backbone = timm.create_model('densenet169', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

def get_model(model_type='ensemble', num_classes=Config.NUM_CLASSES):
    print(f"Creating model: {model_type}")
    
    if model_type == 'ensemble':
        model = ImprovedTransformerEnsembleModel(num_classes, use_aux_loss=True)
        print(f"  ✓ IMPROVED Ensemble with Multi-Scale Fusion + Deep Supervision")
        return model
    
    elif model_type == 'ensemble_old':
        model = TransformerEnsembleModel(num_classes)
        print(f"  ✓ OLD Ensemble model with ViT + Swin + ConvNeXt")
        return model
    
    elif model_type == 'improved_ensemble':
        model = ImprovedTransformerEnsembleModel(num_classes, use_aux_loss=True)
        print(f"  ✓ Improved ensemble with feature attention")
        return model
    
    elif model_type == 'vit':
        model = SingleModel('vit_base_patch16_224', num_classes)
        print(f"  ✓ ViT-Base model")
        return model
    
    elif model_type == 'swin':
        model = SingleModel('swin_base_patch4_window7_224', num_classes)
        print(f"  ✓ Swin-Base model")
        return model
    
    elif model_type == 'convnext':
        model = SingleModel('convnext_base', num_classes)
        print(f"  ✓ ConvNeXt-Base model")
        return model
    
    elif model_type == 'densenet169':
        model = DenseNet169Model(num_classes)
        print(f"  ✓ DenseNet-169 model")
        return model
    
    elif model_type == 'efficientnet_b4':
        model = SingleModel('efficientnet_b4', num_classes)
        print(f"  ✓ EfficientNet-B4 model")
        return model
    
    elif model_type == 'resnet152':
        model = SingleModel('resnet152', num_classes)
        print(f"  ✓ ResNet-152 model")
        return model
    
    elif model_type == 'vit_base':
        model = SingleModel('vit_base_patch16_224', num_classes)
        print(f"  ✓ ViT-Base model")
        return model
    
    elif model_type == 'swin_base':
        model = SingleModel('swin_base_patch4_window7_224', num_classes)
        print(f"  ✓ Swin-Base model")
        return model
    
    elif model_type == 'convnext_base':
        model = SingleModel('convnext_base', num_classes)
        print(f"  ✓ ConvNeXt-Base model")
        return model
    
    elif model_type == 'dino_vit':
        model = SingleModel('vit_base_patch16_224.dino', num_classes)
        print(f"  ✓ DINO ViT model")
        return model
    
    elif model_type == 'sam_vit':
        model = SingleModel('vit_base_patch16_224.sam', num_classes)
        print(f"  ✓ SAM ViT model")
        return model
    
    elif model_type == 'vmamba_base':
        model = VMambaModel(num_classes)
        print(f"  ✓ VMamba model")
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    print("Testing model creation...")
    
    test_input = torch.randn(2, 3, 224, 224)
    
    model = get_model('ensemble', num_classes=15)
    model.eval()
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"\nModel output shapes:")
    if isinstance(output, dict):
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  Output: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1e6:.1f} MB")