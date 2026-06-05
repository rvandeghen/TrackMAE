# TrackMAE Installation

This project relies on several open-source libraries. We recommend using **Conda** to manage the Python environment and installing the pinned dependencies from `requirements.txt`.

## Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/rvandeghen/TrackMAE.git
cd TrackMAE
```

2. **Create a Conda environment**

```bash
conda create -n trackmae python=3.11 -y
```

3. **Activate the environment**

```bash
conda activate trackmae
```

4. **Install the dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## CLIP Target Weights

CLIP weights are required only when pretraining with
`--target_type clip_vit_b16` or `--target_type clip_vit_l14`. They are not
needed for pixel-target pretraining, fine-tuning, or evaluation.

The TrackMAE CLIP loader expects the following files relative to the repository
root:

```text
clip_weights/
|-- ViT-B-16.pt
`-- ViT-L-14.pt
```

Install the Hugging Face Hub CLI and download both visual encoder checkpoints:

```bash
python -m pip install --upgrade huggingface_hub

hf download rvandeghen/TrackMAE \
  clip_weights/ViT-B-16.pt \
  clip_weights/ViT-L-14.pt \
  --local-dir .
```

To download only the checkpoint required by a specific target:

```bash
# For --target_type clip_vit_b16
hf download rvandeghen/TrackMAE clip_weights/ViT-B-16.pt --local-dir .

# For --target_type clip_vit_l14
hf download rvandeghen/TrackMAE clip_weights/ViT-L-14.pt --local-dir .
```
