import argparse
from typing import Literal

import cv2
import safetensors as st
from tinygrad import Tensor, nn
from transform import image2tensor

from dpt import DPTv2, DPTv2Config

model_configs = {
    "vits": {
        "indermediate_layers": [2, 5, 8, 11],
        "depth": 12,
        "num_heads": 6,
        "embed_dim": 384,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "indermediate_layers": [2, 5, 8, 11],
        "depth": 12,
        "num_heads": 12,
        "embed_dim": 768,
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "indermediate_layers": [4, 11, 17, 23],
        "depth": 24,
        "num_heads": 16,
        "embed_dim": 1024,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def get_config(m_size: Literal["vits", "vitb", "vitl", "vitg"]):
    return DPTv2Config(img_size=518, patch_size=14, in_channels=3, mlp_ratio=4, **model_configs[m_size])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPTv2 model on an input image")

    parser.add_argument("--vit-size", type=str, default="s", choices=["s", "b", "l", "g"], help="vit size")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, help="Path to save the output image")

    args = parser.parse_args()

    model_name = f"vit{args.vit_size}"
    config = get_config(model_name)
    model = DPTv2(config)

    Tensor.no_grad = True
    with st.safe_open(f"weights/{model_name}.safetensors", "numpy") as f:
        tensors = {key: Tensor(f.get_tensor(key)) for key in f.keys()}
        nn.state.load_state_dict(model, tensors, verbose=False, strict=True, consume=True)

    # Load and process the image
    image = cv2.imread(args.input)
    image, (h, w) = image2tensor(image, input_size=config.img_size)

    # Run the model and save the output
    output = model(image)
    output = output.interpolate((h, w), mode="linear", align_corners=True).realize()
    output = output.numpy()[0, 0]

    output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    output = cv2.applyColorMap(output, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(args.output if args.output else f"{model_name}.jpg", output)
