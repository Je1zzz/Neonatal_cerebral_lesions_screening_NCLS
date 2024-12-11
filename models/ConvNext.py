<<<<<<< HEAD
"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Classifier(nn.Module):
    def __init__(self, feat_dim, nb_cls, cos_temp):
        super(Classifier, self).__init__()

        fc = nn.Linear(feat_dim, nb_cls)        
        self.weight = nn.Parameter(fc.weight.t(), requires_grad=True)
        self.bias = nn.Parameter(fc.bias, requires_grad=True)
        self.cos_temp = nn.Parameter(torch.FloatTensor(1).fill_(cos_temp), requires_grad=False)
        self.cosine_similarity = self.apply_cosine

    def get_weight(self):
        return self.weight, self.bias

    def apply_cosine(self, feature, weight, bias):
        feature = F.normalize(feature, p=2, dim=1, eps=1e-12)  # 在第2维度上进行归一化
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)  # 在第1维度上进行归一化

        cls_score = self.cos_temp * (torch.mm(feature, weight))
        return cls_score

    def forward(self, feature):
        weight, bias = self.get_weight()
        cls_score = self.cosine_similarity(feature, weight, bias)
        return cls_score

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-2)  

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = self.softmax(attention_scores)
        weighted_features = torch.matmul(attention_weights, V)
        return weighted_features, attention_weights

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, num_img_classes:int =2 ,num_planes:int = 5, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6, cos_classifier:bool = True,
                 head_init_scale: float = 1., patient_classify = True):
        super().__init__()
        self.patient_classify = patient_classify
        self.cos_classifier = cos_classifier
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.self_attention = SelfAttention(dims[-1])
        if self.patient_classify:
            self.self_attention = nn.MultiheadAttention(embed_dim=dims[-1], num_heads=8,batch_first=True)
            self.cls_token = nn.Parameter(torch.randn(1, dims[-1]), requires_grad=True)
            if self.cos_classifier:
                self.disease_classifier = Classifier(dims[-1],num_classes,cos_temp=8)
            else:
                self.disease_classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=False),
                    nn.Linear(dims[-1], num_classes)
                )

        self.instance_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(dims[-1], num_img_classes)
        )

        self.plane_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(dims[-1], num_planes
            )
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        avg_features = self.avgpool(x).view(x.size(0), -1)
        x = self.norm(x.mean([-2, -1]))

        return x, avg_features

    def forward(self, x: torch.Tensor, bag_length, intermediate_features=False) -> torch.Tensor:
        bl, c, h, w = x.size()
        x_features, avg_features = self.forward_features(x)  # [bl, 768]
        max_length = 6
        instance_outputs = self.instance_classifier(x_features)
        plane_outputs = self.plane_classifier(x_features)
        
        ### 
        if self.patient_classify:
            x_split = torch.split(x_features, bag_length.tolist())
            padded_x_features = []
            attn_mask = []
            for i in range(len(bag_length)):
                actual_length = bag_length[i]
                padding_length = max_length - actual_length      
                if padding_length > 0:
                    padding_tensor = torch.zeros((padding_length, x_features.size(-1)), device=x.device)
                    mask_tensor = torch.ones((padding_length,),dtype=torch.bool, device=x.device) # value of padding feature is 1
                    padded_feature = torch.cat((x_split[i], padding_tensor), dim=0)
                    masks = torch.cat((torch.zeros((actual_length,),dtype=torch.bool,device=x.device),mask_tensor),dim=0) 
                else:
                    padded_feature = x_split[i]
                    masks = torch.zeros((actual_length,),dtype=torch.bool,device=x.device)
                padded_x_features.append(padded_feature)
                attn_mask.append(masks)

            padded_x_features = torch.stack(padded_x_features) #[bs,6,768]
            attn_mask = torch.stack(attn_mask) #[bs,6]

            cls_token = self.cls_token.expand(padded_x_features.size(0), -1, -1)  # [bs,1,768]
            feature = torch.cat((cls_token, padded_x_features), dim=1) # [bs,7,768], [32,7,768]
            attn_mask = torch.cat((torch.zeros_like(attn_mask[:, :1],dtype=torch.bool), attn_mask), dim=1) # [bs,7] add mask of cls_token
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # [bs,1,1,7]
            attn_mask = attn_mask.repeat(1,8,7,1) # [bs,8,7,1]
            attn_mask = attn_mask.view(-1,7,7).bool() # [bs*head_num, 7, 7], [256,7,7]
            attended_features, _ = self.self_attention(feature, feature, feature, attn_mask=attn_mask) 
            bag_feature = attended_features[:,0] # use the first feature
            disease_outputs =self.disease_classifier(bag_feature)
            if intermediate_features:
                return disease_outputs, instance_outputs, plane_outputs, avg_features
            else:
                return disease_outputs, instance_outputs, plane_outputs
        else:
            return instance_outputs, plane_outputs


def convnext_tiny(num_classes: int, num_img_classes, num_planes, patient_classify: bool,cos_classifier:bool):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     num_img_classes= num_img_classes,
                     num_planes=num_planes,
                     patient_classify=patient_classify,
                     cos_classifier=cos_classifier)
    return model


def convnext_small(num_classes: int,num_planes,patient_classify:bool):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     num_planes=num_planes,
                     patient_classify=patient_classify)
    return model


def convnext_base(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model


def convnext_large(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model


def convnext_xlarge(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model

def create_model(name='convnext_tiny', num_classes=1000, num_img_classes=2, num_planes=5, patient_classify=True,cos_classifier=False):
    model_dict = {
        'convnext_tiny': convnext_tiny,
        'convnext_small': convnext_small,
        'convnext_base': convnext_base,
        'convnext_large': convnext_large,
        'convnext_xlarge': convnext_xlarge
    }

    if name in model_dict:
        return model_dict[name](num_classes=num_classes, num_img_classes=num_img_classes, num_planes=num_planes, patient_classify=patient_classify,
                                cos_classifier=cos_classifier)
    else:
        raise ValueError(f"Model name '{name}' is not supported.")



if __name__ == "__main__":
    model_name = 'convnext_tiny'  # 你可以根据需要更改模型名称
    num_classes = 1000
    num_img_classes = 2
    num_planes = 5
    patient_classify = True
    cos_classifier = False

    model = create_model(name=model_name, num_classes=num_classes, num_img_classes=num_img_classes, num_planes=num_planes, patient_classify=patient_classify, cos_classifier=cos_classifier)
=======
"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Classifier(nn.Module):
    def __init__(self, feat_dim, nb_cls, cos_temp):
        super(Classifier, self).__init__()

        fc = nn.Linear(feat_dim, nb_cls)        
        self.weight = nn.Parameter(fc.weight.t(), requires_grad=True)
        self.bias = nn.Parameter(fc.bias, requires_grad=True)
        self.cos_temp = nn.Parameter(torch.FloatTensor(1).fill_(cos_temp), requires_grad=False)
        self.cosine_similarity = self.apply_cosine

    def get_weight(self):
        return self.weight, self.bias

    def apply_cosine(self, feature, weight, bias):
        feature = F.normalize(feature, p=2, dim=1, eps=1e-12)  # 在第2维度上进行归一化
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)  # 在第1维度上进行归一化

        cls_score = self.cos_temp * (torch.mm(feature, weight))
        return cls_score

    def forward(self, feature):
        weight, bias = self.get_weight()
        cls_score = self.cosine_similarity(feature, weight, bias)
        return cls_score

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-2)  

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = self.softmax(attention_scores)
        weighted_features = torch.matmul(attention_weights, V)
        return weighted_features, attention_weights

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, num_img_classes:int =2 ,num_planes:int = 5, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6, cos_classifier:bool = True,
                 head_init_scale: float = 1., patient_classify = True):
        super().__init__()
        self.patient_classify = patient_classify
        self.cos_classifier = cos_classifier
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.self_attention = SelfAttention(dims[-1])
        if self.patient_classify:
            self.self_attention = nn.MultiheadAttention(embed_dim=dims[-1], num_heads=8,batch_first=True)
            self.cls_token = nn.Parameter(torch.randn(1, dims[-1]), requires_grad=True)
            if self.cos_classifier:
                self.disease_classifier = Classifier(dims[-1],num_classes,cos_temp=8)
            else:
                self.disease_classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=False),
                    nn.Linear(dims[-1], num_classes)
                )

        self.instance_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(dims[-1], num_img_classes)
        )

        self.plane_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(dims[-1], num_planes
            )
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        avg_features = self.avgpool(x).view(x.size(0), -1)
        x = self.norm(x.mean([-2, -1]))

        return x, avg_features

    def forward(self, x: torch.Tensor, bag_length, intermediate_features=False) -> torch.Tensor:
        bl, c, h, w = x.size()
        x_features, avg_features = self.forward_features(x)  # [bl, 768]
        max_length = 6
        instance_outputs = self.instance_classifier(x_features)
        plane_outputs = self.plane_classifier(x_features)
        
        ### 
        if self.patient_classify:
            x_split = torch.split(x_features, bag_length.tolist())
            padded_x_features = []
            attn_mask = []
            for i in range(len(bag_length)):
                actual_length = bag_length[i]
                padding_length = max_length - actual_length      
                if padding_length > 0:
                    padding_tensor = torch.zeros((padding_length, x_features.size(-1)), device=x.device)
                    mask_tensor = torch.ones((padding_length,),dtype=torch.bool, device=x.device) # value of padding feature is 1
                    padded_feature = torch.cat((x_split[i], padding_tensor), dim=0)
                    masks = torch.cat((torch.zeros((actual_length,),dtype=torch.bool,device=x.device),mask_tensor),dim=0) 
                else:
                    padded_feature = x_split[i]
                    masks = torch.zeros((actual_length,),dtype=torch.bool,device=x.device)
                padded_x_features.append(padded_feature)
                attn_mask.append(masks)

            padded_x_features = torch.stack(padded_x_features) #[bs,6,768]
            attn_mask = torch.stack(attn_mask) #[bs,6]

            cls_token = self.cls_token.expand(padded_x_features.size(0), -1, -1)  # [bs,1,768]
            feature = torch.cat((cls_token, padded_x_features), dim=1) # [bs,7,768], [32,7,768]
            attn_mask = torch.cat((torch.zeros_like(attn_mask[:, :1],dtype=torch.bool), attn_mask), dim=1) # [bs,7] add mask of cls_token
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # [bs,1,1,7]
            attn_mask = attn_mask.repeat(1,8,7,1) # [bs,8,7,1]
            attn_mask = attn_mask.view(-1,7,7).bool() # [bs*head_num, 7, 7], [256,7,7]
            attended_features, _ = self.self_attention(feature, feature, feature, attn_mask=attn_mask) 
            bag_feature = attended_features[:,0] # use the first feature
            disease_outputs =self.disease_classifier(bag_feature)
            if intermediate_features:
                return disease_outputs, instance_outputs, plane_outputs, avg_features
            else:
                return disease_outputs, instance_outputs, plane_outputs
        else:
            return instance_outputs, plane_outputs


def convnext_tiny(num_classes: int, num_img_classes, num_planes, patient_classify: bool,cos_classifier:bool):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     num_img_classes= num_img_classes,
                     num_planes=num_planes,
                     patient_classify=patient_classify,
                     cos_classifier=cos_classifier)
    return model


def convnext_small(num_classes: int,num_planes,patient_classify:bool):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     num_planes=num_planes,
                     patient_classify=patient_classify)
    return model


def convnext_base(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model


def convnext_large(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model


def convnext_xlarge(num_classes: int,num_planes):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     num_planes=num_planes)
    return model

def create_model(name='convnext_tiny', num_classes=1000, num_img_classes=2, num_planes=5, patient_classify=True,cos_classifier=False):
    model_dict = {
        'convnext_tiny': convnext_tiny,
        'convnext_small': convnext_small,
        'convnext_base': convnext_base,
        'convnext_large': convnext_large,
        'convnext_xlarge': convnext_xlarge
    }

    if name in model_dict:
        return model_dict[name](num_classes=num_classes, num_img_classes=num_img_classes, num_planes=num_planes, patient_classify=patient_classify,
                                cos_classifier=cos_classifier)
    else:
        raise ValueError(f"Model name '{name}' is not supported.")



if __name__ == "__main__":
    model_name = 'convnext_tiny'  # 你可以根据需要更改模型名称
    num_classes = 1000
    num_img_classes = 2
    num_planes = 5
    patient_classify = True
    cos_classifier = False

    model = create_model(name=model_name, num_classes=num_classes, num_img_classes=num_img_classes, num_planes=num_planes, patient_classify=patient_classify, cos_classifier=cos_classifier)
>>>>>>> origin/main
    print(model)