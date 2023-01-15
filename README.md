# Simple Dedup

simple dedup based on perceptual hashes from <https://phash.org> and using FAISS to construct indexes.

## Install

## 1) Install Phash 

First, you need to install a fork Phash library (https://github.com/mehdidc/pHash), which contains a simple extension to compute hashes
from buffers instead of files, for more efficiency.

## 2) Install FAISS

Instructions: <https://github.com/facebookresearch/faiss>

## 3) Install requirements

`pip install -r requirements.txt`

## Usage

### 1) Compute Perceptual hashes

- On WDS datasets:
    
    `python cli.py compute-hashes wds "<path>/{00000..41455}.tar" --batch-size=10000 --workers=64 --out-path=hashes_upstream.npz`

- On datasets supported by CLIP benchmark (https://github.com/LAION-AI/CLIP_benchmark):
    
    `python cli.py compute-hashes imagenet1k <path_root_imagenet1k> --batch-size=10000 --workers=64 --out-path=hashes_imagenet1k.npz`

### 2) Build index

`python cli.py build-index hashes_upstream.npz --out-index=index_upstream.pkl  --out-meta=index_meta.pkl`

### 3) Find duplicates

`python cli.py dupfind index_upstream.pkl hashes_imagenet1k.npz --threshold=1 --out-path=dups_imagenet1k.csv`

### 4) Visualize duplicates

`python cli.py build-html-visualizer index_meta.pkl  dups_imagenet1k.csv imagenet1k <path_root_imagenet1k>`
