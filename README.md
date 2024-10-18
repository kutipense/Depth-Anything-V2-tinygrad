# Depth-Anything-V2 Tinygrad

This is a single-file `tinygrad` implementation of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

## Installation

To install the required dependencies, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Weights

Please refer to Hugging Face for the required model weights. After downloading the `.safetensors` files, place them under the `weights/` directory. The filenames should follow this pattern:

- `weights/vits.safetensors`
- `weights/vitb.safetensors`
- `weights/vitl.safetensors`

| Model | Download Link                                                                |
| ----- | ---------------------------------------------------------------------------- |
| Small | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| Base  | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf)  |
| Large | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) |

## Usage

```bash
usage: main.py [-h] [--vit-size {s,b,l,g}] --input INPUT
               [--output OUTPUT]

Run DPTv2 model on an input image.

options:
  -h, --help            Show this help message and exit.
  --vit-size {s,b,l,g}  Specify the Vision Transformer size (`s`, `b`, `l`, `g`).
  --input INPUT         Path to the input image.
  --output OUTPUT       Path to save the output image.
```

### Example commands

- To display the output image using `matplotlib`:

  ```bash
  python main.py --vit-size s --input docs/input.jpg
  ```

- To save the output to a file:

  ```bash
  python main.py --vit-size l --input docs/input.jpg --output docs/output.jpg
  ```

## Example Images

### Input

![Input Image](docs/input.jpg)

### Output

![Output Image (ViT-L)](docs/vitl.jpg)

---

This README was partially written with the assistance of AI.
