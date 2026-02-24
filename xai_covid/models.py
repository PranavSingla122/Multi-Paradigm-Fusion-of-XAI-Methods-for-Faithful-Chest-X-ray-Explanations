import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing."""
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs).scatter_(
                    1, targets.unsqueeze(1), 1.0
                )
                smooth_targets = (
                    smooth_targets * (1 - self.label_smoothing)
                    + self.label_smoothing / num_classes
                )
            log_prob = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_targets * log_prob).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, weight=self.weight, reduction='none'
            )

        pt = torch.exp(
            -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        )
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1-D feature vectors."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x.unsqueeze(2)).squeeze(2)
        y = self.excitation(y)
        return x * y


class ChannelAttention(nn.Module):
    """Dual-pool channel attention for 1-D feature vectors."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x.unsqueeze(2)).squeeze(2))
        max_out = self.fc(self.max_pool(x.unsqueeze(2)).squeeze(2))
        return x * self.sigmoid(avg_out + max_out)


class MultiScaleFeatureFusion(nn.Module):

    def __init__(self, vit_dim, swin_dim, convnext_dim, output_dim=1024):
        super().__init__()

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        self.swin_proj = nn.Sequential(
            nn.Linear(swin_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        self.convnext_proj = nn.Sequential(
            nn.Linear(convnext_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        self.vit_channel_att      = ChannelAttention(output_dim)
        self.swin_channel_att     = ChannelAttention(output_dim)
        self.convnext_channel_att = ChannelAttention(output_dim)

        self.gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 3),
            nn.Sigmoid(),
        )

    def forward(self, vit_feat, swin_feat, convnext_feat):
        vit_p      = self.vit_channel_att(self.vit_proj(vit_feat))
        swin_p     = self.swin_channel_att(self.swin_proj(swin_feat))
        convnext_p = self.convnext_channel_att(self.convnext_proj(convnext_feat))

        stacked = torch.stack([vit_p, swin_p, convnext_p], dim=1)   # (B, 3, D)
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        concat = torch.cat([attended[:, 0], attended[:, 1], attended[:, 2]], dim=1)
        return concat * self.gate(concat)                            # (B, 3*D)

class ImprovedTransformerEnsembleModel(nn.Module):

    def __init__(self, num_classes: int = Config.NUM_CLASSES, use_aux_loss: bool = True):
        super().__init__()
        self.num_classes  = num_classes
        self.use_aux_loss = use_aux_loss

        # ── Backbones ─────────────────────────────────────────────────────────
        self.vit      = timm.create_model('vit_base_patch16_224',         pretrained=True, num_classes=0)
        self.swin     = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.convnext = timm.create_model('convnext_base',                pretrained=True, num_classes=0)

        vit_dim      = self.vit.num_features       # 768
        swin_dim     = self.swin.num_features      # 1024
        convnext_dim = self.convnext.num_features  # 1024

        # ── Feature fusion ────────────────────────────────────────────────────
        self.feature_fusion = MultiScaleFeatureFusion(
            vit_dim, swin_dim, convnext_dim, output_dim=1024
        )

        # ── Ensemble head ─────────────────────────────────────────────────────
        self.ensemble_head = nn.Sequential(
            nn.Linear(1024 * 3, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024),     nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),      nn.LayerNorm(512),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, num_classes),
        )

        # ── Auxiliary heads (deep supervision) ───────────────────────────────
        if self.use_aux_loss:
            self.vit_aux      = self._make_aux_head(vit_dim,      num_classes)
            self.swin_aux     = self._make_aux_head(swin_dim,     num_classes)
            self.convnext_aux = self._make_aux_head(convnext_dim, num_classes)

        # ── Learnable temperature ─────────────────────────────────────────────
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _make_aux_head(self, in_features: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256),         nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        vit_f      = self.vit(x)
        swin_f     = self.swin(x)
        convnext_f = self.convnext(x)

        fused  = self.feature_fusion(vit_f, swin_f, convnext_f)
        logits = self.ensemble_head(fused) / self.temperature

        out = {'logits': logits, 'ensemble': logits}

        if self.use_aux_loss and self.training:
            out['vit_aux']      = self.vit_aux(vit_f)
            out['swin_aux']     = self.swin_aux(swin_f)
            out['convnext_aux'] = self.convnext_aux(convnext_f)

        if return_features:
            out['features']          = fused
            out['vit_features']      = vit_f
            out['swin_features']     = swin_f
            out['convnext_features'] = convnext_f

        return out

class TransformerEnsembleModel(nn.Module):
    """Original model kept for ablation studies. NOT recommended for training."""
    def __init__(self, num_classes: int = Config.NUM_CLASSES):
        super().__init__()
        self.vit_base      = timm.create_model('vit_base_patch16_224',         pretrained=False, num_classes=0)
        self.swin_base     = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        self.convnext_base = timm.create_model('convnext_base',                pretrained=False, num_classes=0)

        vit_f = self.vit_base.num_features
        sw_f  = self.swin_base.num_features
        cn_f  = self.convnext_base.num_features

        self.vit_head      = self._make_head(vit_f, num_classes)
        self.swin_head     = self._make_head(sw_f,  num_classes)
        self.convnext_head = self._make_head(cn_f,  num_classes)

        self.ensemble_layer = nn.Sequential(
            nn.Linear(vit_f + sw_f + cn_f, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(1024, 512),                  nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
        )
        self.se_block   = SEBlock(512)
        self.classifier = nn.Linear(512, num_classes)
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)

    def _make_head(self, in_f, num_classes):
        return nn.Sequential(
            nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        vf = self.vit_base(x);      vo = self.vit_head(vf)
        sf = self.swin_base(x);     so = self.swin_head(sf)
        cf = self.convnext_base(x); co = self.convnext_head(cf)

        fused = self.ensemble_layer(torch.cat([vf, sf, cf], dim=1))
        fused = self.se_block(fused)
        ens   = self.classifier(fused)

        w     = F.softmax(self.attention_weights, dim=0)
        final = 0.6 * ens + 0.4 * (vo * w[0] + so * w[1] + co * w[2])

        return {
            'ensemble': final, 'vit_base': vo,
            'swin_base': so,   'convnext_base': co,
            'attention_weights': w, 'features': fused,
        }

class MCDropoutModel(nn.Module):
    def __init__(self, base_model, dropout_rate: float = Config.DROPOUT_RATE):
        super().__init__()
        self.base_model = base_model
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, x, mc_samples: int = Config.MC_SAMPLES):
        if self.training:
            return self.base_model(x)
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
        return predictions.mean(dim=0), predictions.var(dim=0).mean(dim=1)


class SingleModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224',
                 num_classes: int = Config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        nf = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(nf, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class DenseNet169Model(nn.Module):
    def __init__(self, num_classes: int = Config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('densenet169', pretrained=pretrained, num_classes=0)
        nf = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(nf, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class VMambaModel(nn.Module):
    def __init__(self, num_classes: int = Config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            self.model = timm.create_model('vmamba_base',   pretrained=pretrained, num_classes=num_classes)
        except Exception:
            self.model = timm.create_model('vssm_base_v2', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(model_type: str = 'ensemble', num_classes: int = Config.NUM_CLASSES):
    print(f"Creating model: {model_type}")

    dispatch = {
        'ensemble':          lambda: ImprovedTransformerEnsembleModel(num_classes, use_aux_loss=True),
        'improved_ensemble': lambda: ImprovedTransformerEnsembleModel(num_classes, use_aux_loss=True),
        'ensemble_old':      lambda: TransformerEnsembleModel(num_classes),
        'vit':               lambda: SingleModel('vit_base_patch16_224', num_classes),
        'vit_base':          lambda: SingleModel('vit_base_patch16_224', num_classes),
        'swin':              lambda: SingleModel('swin_base_patch4_window7_224', num_classes),
        'swin_base':         lambda: SingleModel('swin_base_patch4_window7_224', num_classes),
        'convnext':          lambda: SingleModel('convnext_base', num_classes),
        'convnext_base':     lambda: SingleModel('convnext_base', num_classes),
        'densenet169':       lambda: DenseNet169Model(num_classes),
        'efficientnet_b4':   lambda: SingleModel('efficientnet_b4', num_classes),
        'resnet152':         lambda: SingleModel('resnet152', num_classes),
        'dino_vit':          lambda: SingleModel('vit_base_patch16_224.dino', num_classes),
        'sam_vit':           lambda: SingleModel('vit_base_patch16_224.sam', num_classes),
        'vmamba_base':       lambda: VMambaModel(num_classes),
    }

    if model_type not in dispatch:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Choose from: {list(dispatch.keys())}"
        )

    model     = dispatch[model_type]()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,} | Trainable: {trainable:,} | Size: {total*4/1e6:.1f} MB")
    return model

if __name__ == "__main__":
    print("Testing COVID model creation...")
    x = torch.randn(2, 3, 224, 224)

    model = get_model('ensemble', num_classes=Config.NUM_CLASSES)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print("\nOutput shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")