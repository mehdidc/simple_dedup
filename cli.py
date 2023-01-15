import io
from io import BytesIO
import base64
import time
import pandas as pd
from collections import defaultdict
import json
from pathlib import Path
from glob import glob
import os
import shutil
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
import webdataset as wds
import cv2

from phash import DCTImageHash, MHImageHash
from resizer import Resizer
from clize import run

resizers = {
    # we apply the same resize function as upstream dataset on downstream dataset
    "laion400m": Resizer(
        image_size=256,
        resize_mode="border",
        resize_only_if_bigger=False,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        encode_format="jpg",
        skip_reencode=False,
        disable_all_reencoding=False,
        min_image_size=0,
        max_aspect_ratio=float("inf"),
    )
}

def log_and_continue(exn):
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True
def dct_hash_image_from_buffer(buf):
    return DCTImageHash.from_buffer(buf)
def mhi_hash_image_from_buffer(buf):
    return MHImageHash.from_buffer(buf)
hash_functions = [
    dct_hash_image_from_buffer,
    # mhi_hash_image_from_buffer,
]
phash_human_name = {
   "dct_hash_image_from_buffer": "phash_dct",
}
hash_names = tuple([phash_human_name[h.__name__] for h in hash_functions])
def compute_hashes_from_iterator(iterator):
    for data in iterator:
        success = True
        for compute_hash in hash_functions:
            name = compute_hash.__name__
            try:
                value = compute_hash(data["image"]).get_hash()
            except Exception:
                value = 0
                success = False
            data[phash_human_name[name]] = value
        if "json" in data:
            js = json.loads(data["json"])
            data["url_caption_hash"] = hash(str(js['caption']) + str(js['url']))
            data["url"] = js['url']
        else:
            data["url_caption_hash"] = 0
            data['url'] = ''
        data['success'] = success
        data['index'] = 0
        yield data

class Filter:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        return any(k in sample for k in self.keys)

def compute_hashes(
    dataset_name, dataset_root, *, 
    batch_size=10_000, 
    batches_per_chunk:int=None,
    workers=64,
    image_key="jpg;png",  # for webdataset
    resizer="laion400m", 
    split="test", 
    out_prefix="",
    out_path="hashes.npz",
):
    """
    Compute hashes on a dataset
    support Webdataset and datasets from CLIP benchmark

    dataset_name: wds or any CLIP benchmark dataset
    dataset_root: for wds, tar files. for CLIP benchmark, the dataset root
    
    batches_per_chunk: int
        nb of batches to store per chunk, each chunk will contain batch_size*batches_per_chunk.
        if None, not used, equivalent to having one chunk which contains everything
    """
    if dataset_name == "wds":
        pipeline = [wds.SimpleShardList(dataset_root)]
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.select(Filter(image_key.split(";"))),
        ])
        keys = ("index", "url_caption_hash", "url", "success") + hash_names
        pipeline.extend([
            wds.rename(image=image_key),
            compute_hashes_from_iterator,
            wds.to_tuple( *( keys ) ),
            wds.batched(batch_size),
        ])
        dataset = wds.DataPipeline(*pipeline)
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
        )
    else:
        from clip_benchmark.datasets.builder import (
            build_dataset,
            get_dataset_collate_fn,
        )
        from functools import partial
        import torch
        resizer = resizers[resizer] if resizer else None
        dataset = DatasetWithHashWrapper(build_dataset(
            dataset_name=dataset_name,
            root=dataset_root,
            download=True,
            split=split,
            transform=None,
        ), resizer=resizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=lambda x:list(zip(*x))
        )
        keys = ("index","success") + hash_names
    hashes_dict = defaultdict(list)
    nb = 0
    start = time.time()
    batches = 0
    chunk = 0
    
    for batch in dataloader:
        for name, data in zip(keys, batch):
            hashes_dict[name].extend(data)
        elapsed = time.time() - start
        nb += len(batch[0])
        print(nb, "processed")
        batches += 1
        if batches_per_chunk is not None and batches % batches_per_chunk == 0:
            print("Elapsed", elapsed, "impersec", nb/elapsed, "secperim", elapsed/nb)
            cast_uint64_(hashes_dict, hash_names)
            if out_prefix:
                np.savez(out_prefix + "_" + str(chunk), **hashes_dict)
            hashes_dict = defaultdict(list)
            chunk += 1
    if len(hashes_dict):
        cast_uint64_(hashes_dict, hash_names)
        if out_prefix:
            np.savez(out_prefix + "_" + str(chunk), **hashes_dict)
        else:
            np.savez(out_path, **hashes_dict)
    print("Elapsed", elapsed, "impersec", nb/elapsed, "secperim", elapsed/nb)
    print("FINISHED")

def build_index(pattern, *, out_index="index.pkl", out_meta="index_meta.pkl", dim_bits=64, key="phash_dct", index_type="binary_hash"):
    """
    Create an index from computed hashes (computed using `compute_hashes`) 
    Can be used to dedup
    """
    import joblib
    import faiss
    from glob import glob
    if index_type == "flat":
        index = faiss.IndexBinaryFlat(dim)
    elif index_type == "binary_hash":
        index = faiss.IndexBinaryHash(dim_bits, 10)
    else:
        raise ValueError(f"{index_type} should be `flat` or `binary_hash`")
    urls = []
    for path in sorted(glob(pattern)):
        if path.endswith(".npz"):
            data = np.load(path)
            url = data["url"]
            hashes = data[key]
        elif path.endswith(".csv"):
            data = pd.read_csv(path)
            hashes = data.phash.values
            url = data.url.values.tolist()
        else:
            continue
        print(len(hashes), len(url))
        hashes = uint64_to_uint8_vec(hashes)
        urls.extend(url)
        index.add(hashes)
    faiss.write_index_binary(index, out)
    joblib.dump({"url": urls}, out_meta)

    
class DatasetWithHashWrapper:
    def __init__(self, ds, resizer=None):
        self.ds = ds
        self.resizer = resizer

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        vals = []
        success = True
        for compute_hash in hash_functions:
            name = compute_hash.__name__
            try:
                fd = io.BytesIO()
                image.save(fd, format='jpeg')
                if self.resizer is not None:
                    image_bytes, width, height, original_width, original_height, _ = self.resizer(fd)
                    if image_bytes is None:
                        raise Exception("exception")
                else:
                    image_bytes = fd.getvalue()
                value = compute_hash(image_bytes).get_hash()
            except Exception as ex:
                value = 0
                success = False
            vals.append(value)
        return (idx, success) + tuple(vals)

    def __len__(self):
        return len(self.ds)

def uint64_to_uint8_vec(dst):
    dst = [f"{int(l):016x}" for l in dst] # To Hex
    dst = np.array([hex_to_uint8_vec(s) for s in dst])
    return dst


def dupfind(upstream_index_path, downstream_hashes_path, threshold=1, out_path="dups.csv", key="phash_dct"):
    """
    Find duplicates on a dataset based on computed hashes (`compute_hashes`) and using an index (`build_index`)

    upstream_index_path: str
        path to index (output of `build_index`)

    downstream_hashes_path: str
        path to downstream hashes (computed using `compute_hashes`)

    threshold: int
        threshold to use for hamming distance.
        that is, any pairs with distance strictly smaller than the threshold
        will be considered duplicates.
    """
    import faiss
    import joblib
    src = faiss.read_index_binary(upstream_index_path)
    dst = np.load(downstream_hashes_path)[key]
    dst = uint64_to_uint8_vec(dst)
    print("Doing Range Search")
    L, D, I = src.range_search(dst, threshold)
    rows = []
    nb = 0
    for i in range(len(L) - 1):
        indices = I[L[i] : L[i + 1]]
        dists = D[L[i] : L[i + 1]]
        for d, ind in zip(dists, indices):
            rows.append({"index_upstream": ind , "index_downstream": i, "dist": d})
        if len(indices):
            nb += 1
    print("NB dups:", nb)
    print("Write dups to", out_path)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def build_html_visualizer(
    upstream_meta_path, 
    downstream_dups_path, 
    dataset_name, dataset_root, *, 
    split="test", 
    out="out.html", per_row=5, 
    cache_folder="cache", 
    imresize=224,
):
    """
    Create an HTML static page with all dups, for visualization
    """
    import joblib
    import PIL
    from subprocess import call
    from clip_benchmark.datasets.builder import (
	build_dataset,
	get_dataset_collate_fn,
    )
    import hashlib
    import faiss
    from tqdm import tqdm

    print("Load upstream urls")
    src_urls = np.array(joblib.load(upstream_meta_path)["url"])
    print("Build downstream dataset")
    ds = build_dataset(
        dataset_name=dataset_name,
        root=dataset_root,
        download=True,
        split=split,
        transform=None,
        annotation_file="",
    )
    N = 0
    lines = []
    invalid = set()
    dst = pd.read_csv(downstream_dups_path)
    dst = dst.sort_values(by="index_downstream")
    for ind in tqdm(dst.index_downstream.unique()):
        upstream_indices = dst[dst.index_downstream==ind].index_upstream.values
        urls = src_urls[upstream_indices] # get urls from upstream
        img, label = ds[ind] # get the image from downstream
        img = (img.resize((imresize, imresize)))
        by = BytesIO()
        img.save(by, format='jpeg')
        imgdata = by.getvalue()
        l = [to_img_tag(imgdata)]
        urls = list(sorted(list(set(urls))))
        urls = urls[:per_row] # only show few
        for url in urls:
            uid = hashlib.md5(url.encode()).hexdigest()
            path = f"{cache_folder}/{uid}.jpg"
            if path in invalid:
                continue
            if not os.path.exists(path):
                call(f"wget '{url}' --output-document={path} --timeout=10 --tries=5", shell=True)
            N += 1
            try:
                img = PIL.Image.open(path).convert('RGB').resize((imresize, imresize))
            except Exception as ex:
                print(ex)
                invalid.add(path)
                continue
            by = BytesIO()
            img.save(by, format='jpeg')
            imgdata = by.getvalue()                
            l.append(to_img_tag(imgdata))
        lines.append(l)
    table = "\n".join("<tr>" + ("".join([f"<td>{l}</td" for l in line])) + f"</tr>" for i, line in enumerate(lines))
    total = len(lines)
    print("Write html to", out)
    with open(out, "w") as fd:
        fd.write("<html><table>" + table + f"</table><p>Total: {total}</p><html>")

def to_img_tag(data):
    data_base64 = base64.b64encode(data)
    data_base64 = data_base64.decode()
    html = '<img src="data:image/jpeg;base64,' + data_base64 + '">'
    return html

def hex_to_uint8_vec(s):
    vals = []
    for i in range(0, len(s), 2):
        vals.append(int(s[i:i+2], 16))
    return np.array(vals, dtype="uint8")

def cast_uint64_(x, keys):
    for k in keys:
        x[k] = np.array(x[k], dtype="uint64")
    return x

if __name__ == "__main__":
    run([compute_hashes, dupfind, build_index, build_html_visualizer])
