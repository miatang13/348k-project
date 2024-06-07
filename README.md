# Stroke Labeling Codebase

## Install Environment

### Create env

```bash
conda create -n stroke python=3.10 -y
conda activate stroke
```

### Install torch
```bash
pip3 install torch torchvision torchaudio
```
### Install diffusers and other dependencies

```bash
pip install -U git+https://github.com/huggingface/diffusers.git
pip install -U controlnet_aux==0.0.7
pip install transformers accelerate safetensors
```

### Some misc utils

```bash
pip install -r requirements.txt
```

### Set up SAM

```bash
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

# download the pretrained model
cd data/ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # all pth are gitignored
```

### Set up grounded SAM
This can get very complicated. But follow https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main 
with the sanity checks:
- Check gcc and g++ version are 9 
- Downgrade setuptools with `python -m pip install setuptools==69.5.1`
- Need to update setuptools for RAM