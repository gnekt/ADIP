# ADIP

# Installation

Open your terminal and type
```bash
    git clone https://github.com/gnekt/ADIP.git
    cd ADIP
    git clone https://github.com/open-mmlab/mmaction2.git
```

Install a python interpreter, version 3.7.13, next install these packages:
```bash
    pip3 --no-cache-dir install torch==1.9.0+cu(*CUDA_VER*) torchvision==0.10.0+cu(*CUDA_VER*) torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

    cd mmaction2

    pip install -e .
    pip install -r requirements/optional.txt

    cd ..
```