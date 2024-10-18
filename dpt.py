from dataclasses import dataclass

from tinygrad import Tensor, nn


@dataclass
class DPTv2Config:
    img_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    mlp_ratio: int
    num_heads: int
    features: int
    out_channels: list[int]
    indermediate_layers: list[int]


class PatchEmbeddings:
    def __init__(self, config: DPTv2Config):
        self.projection = nn.Conv2d(
            config.in_channels, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class Embeddings:
    def __init__(self, config: DPTv2Config, num_tokens=1):
        num_patches = (config.img_size // config.patch_size) ** 2

        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = Tensor.zeros(1, 1, config.embed_dim)
        self.mask_token = Tensor.zeros(1, config.embed_dim)  # unused
        self.position_embeddings = Tensor.zeros(1, num_patches + num_tokens, config.embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.patch_embeddings(x)
        x = Tensor.cat(self.cls_token.expand(x.shape[0], -1, -1), x, dim=1)
        x = x + self.position_embeddings

        return x


class Attention:
    def __init__(self, config: DPTv2Config):
        self.num_heads = config.num_heads
        self.scale = (config.embed_dim // config.num_heads) ** -0.5

        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        ch = C // self.num_heads
        q = self.query(x).reshape(B, N, self.num_heads, ch).transpose(2, 1)
        k = self.key(x).reshape(B, N, self.num_heads, ch).transpose(2, 1)
        v = self.value(x).reshape(B, N, self.num_heads, ch).transpose(2, 1)

        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(axis=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return x


class MLP:
    def __init__(self, config: DPTv2Config):
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).gelu())


class Layer:
    def __init__(self, config: DPTv2Config):
        self.attention = Attention(config)
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.layer_scales = [Tensor.ones(config.embed_dim) * 1e-5 for _ in range(2)]
        self.norms = [nn.LayerNorm(config.embed_dim, eps=1e-6) for _ in range(2)]
        self.mlp = MLP(config)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.layer_scales[0] * self.dense(self.attention(self.norms[0](x)))
        x = x + self.layer_scales[1] * self.mlp(self.norms[1](x))
        return x

    def _asdict(self):
        return {
            "attention.attention": self.attention,
            "attention.output.dense": self.dense,
            "layer_scale1.lambda1": self.layer_scales[0],
            "layer_scale2.lambda1": self.layer_scales[1],
            "mlp": self.mlp,
            "norm1": self.norms[0],
            "norm2": self.norms[1],
        }


class Encoder:
    def __init__(self, config: DPTv2Config):
        self.layer = [Layer(config) for _ in range(config.depth)]

    def __call__(self, x: Tensor) -> Tensor:
        outputs = []
        for layer in self.layer:
            x = layer(x)
            outputs.append(x)
        return outputs


class Backbone:
    def __init__(self, config: DPTv2Config):
        self.indermediate_layers = config.indermediate_layers
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.layernorm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.encoder(self.embeddings(x))
        return [self.layernorm(x[ind]) for ind in self.indermediate_layers]


class Head:
    def __init__(self, config: DPTv2Config):
        in_feats, out_feats = config.features, config.features // 2
        self.conv1 = nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_feats, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)

        self.patch_h = self.patch_w = config.img_size // config.patch_size
        self.patch_h = self.patch_w = self.patch_h * config.patch_size

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).interpolate((self.patch_h, self.patch_w), align_corners=True)
        x = self.conv3(self.conv2(x).relu()).relu()
        return x


class ResidualLayer:
    def __init__(self, config: DPTv2Config):
        in_feats = config.features
        self.convolution1 = nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1)
        self.convolution2 = nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.convolution2(self.convolution1(x.relu()).relu()) + x


class FusionStage:
    def __init__(self, config: DPTv2Config):
        in_feats = config.features
        self.residual_layer1 = ResidualLayer(config)
        self.residual_layer2 = ResidualLayer(config)
        self.projection = nn.Conv2d(in_feats, in_feats, kernel_size=1)

    def __call__(self, layer0: Tensor, layer1: Tensor = None, size=None) -> Tensor:
        if layer1 is not None:
            layer0 = layer0 + self.residual_layer1(layer1)

        layer0 = self.residual_layer2(layer0)
        size = list(map(lambda x: x * 2, layer0.shape[2:])) if size is None else size
        return self.projection(layer0.interpolate(size, align_corners=True))


class ReassembleStage:
    def __init__(self, config: DPTv2Config):
        ins, outs = config.embed_dim, config.out_channels

        self.projection = [
            nn.Conv2d(in_channels=ins, out_channels=out_channel, kernel_size=1) for out_channel in outs
        ]

        self.resize_layers = [
            nn.ConvTranspose2d(in_channels=outs[0], out_channels=outs[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(in_channels=outs[1], out_channels=outs[1], kernel_size=2, stride=2),
            lambda x: x,
            nn.Conv2d(in_channels=outs[3], out_channels=outs[3], kernel_size=3, stride=2, padding=1),
        ]

        self.patch_h = self.patch_w = config.img_size // config.patch_size

    def __call__(self, inputs: list[Tensor]) -> list[Tensor]:
        outputs = []
        for i, out in enumerate(inputs):
            x = out[:, 1:]  # remove the cls token
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], self.patch_h, self.patch_w))
            x = self.resize_layers[i](self.projection[i](x))
            outputs.append(x)
        return outputs

    def _asdict(self):
        return {
            "layers": [{"projection": p, "resize": r} for p, r in zip(self.projection, self.resize_layers)]
        }


class Neck:
    def __init__(self, config: DPTv2Config):
        self.convs = [
            nn.Conv2d(in_channels=ch, out_channels=config.features, kernel_size=3, padding=1, bias=False)
            for ch in config.out_channels
        ]

        self.reassemble_stage = ReassembleStage(config)
        self.fusion_stage = [FusionStage(config) for _ in range(4)]

    def __call__(self, x: Tensor) -> Tensor:
        outputs = self.reassemble_stage(x)
        outputs = [conv(out) for out, conv in zip(outputs, self.convs)]

        path_4 = self.fusion_stage[0](outputs[3], size=outputs[2].shape[2:])
        path_3 = self.fusion_stage[1](path_4, outputs[2], size=outputs[1].shape[2:])
        path_2 = self.fusion_stage[2](path_3, outputs[1], size=outputs[0].shape[2:])
        path_1 = self.fusion_stage[3](path_2, outputs[0])

        return path_1

    def _asdict(self):
        return {
            "convs": self.convs,
            "fusion_stage.layers": self.fusion_stage,
            "reassemble_stage": self.reassemble_stage,
        }


class DPTv2:
    def __init__(self, config):
        self.backbone = Backbone(config)
        self.head = Head(config)
        self.neck = Neck(config)

    def __call__(self, x: Tensor) -> Tensor:
        return self.head(self.neck(self.backbone(x)))
