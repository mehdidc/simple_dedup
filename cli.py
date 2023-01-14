import io
from io import BytesIO
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from clize import run
import time
from PIL import Image
import json
from pathlib import Path
from glob import glob
import os
import shutil
import webdataset as wds
import logging
from multiprocessing import Pool, cpu_count
from phash import DCTImageHash, MHImageHash
import cv2
from resizer import Resizer

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
hash_names = tuple([h.__name__ for h in hash_functions])
def compute_hashes(iterator):
    for data in iterator:
        for compute_hash in hash_functions:
            name = compute_hash.__name__
            try:
                # check_image(data["image"])
                value = compute_hash(data["image"])
                value_int = value.get_hash()
            except Exception:
                value = ''
                value_int = -1
                value_str = ""
            else:
                value_str = f"{value_int:x}"
            data[name] = value
            data[name + "_str"] = value_str
        if "json" in data:
            js = json.loads(data["json"])
            data["url_caption_hash"] = hash(str(js['caption']) + str(js['url']))
        else:
            data["url_caption_hash"] = 0
        yield data

def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)

def upstream(dataset_path, out_path, *, batch_size=10_000, workers=64, image_key="jpg;png", caption_key="json", steps_per_chunk=10000):
    """
    Compute upstream Webdatasegt based (LAION-400M, LAION-2B) hashes
    """
    print(dataset_path, out_path)
    pipeline = [wds.SimpleShardList(dataset_path)]
    pipeline.extend([
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
    ])
    hash_names_str = tuple([h + "_str" for h in hash_names])
    hash_names_all = hash_names + hash_names_str
    pipeline.extend([
        # wds.decode("rgb8", handler=log_and_continue),
        wds.rename(image=image_key),
        compute_hashes,
        wds.to_tuple( *( ("url_caption_hash",) + hash_names_all) ),
        wds.batched(batch_size),
    ])
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
    )
    hashes_dict = defaultdict(list)
    metas = []
    nb = 0
    start = time.time()
    steps = 0
    chunk = 0

    def get_res(hashes_dict, metas):
        res = {}
        for h in hash_names:
            v = hashes_dict[h]
            v = np.concatenate(v)
            res[h] = v
        for h in hash_names_str:
            res[h] = hashes_dict[h]
        res["url_caption_hash"] = metas
        return res

    for meta, *hashes in dataloader:
        for name, hs in zip(hash_names, hashes[:len(hash_names)] ):
            hashes_dict[name].append(hs)
        for name, hs in zip(hash_names_str, hashes[len(hash_names):]):
            hashes_dict[name].extend(hs)
        metas.extend(meta)
        elapsed = time.time() - start
        nb += len(meta)
        steps += 1
        if steps % steps_per_chunk == 0:
            print("Elapsed", elapsed, "impersec", nb/elapsed, "secperim", elapsed/nb)
            res = get_res(hashes_dict, metas)
            np.savez(out_path + "_" + str(chunk), **res)
            hashes_dict = defaultdict(list)
            metas = []
            chunk += 1
    if len(metas):
        res = get_res(hashes_dict, metas)
        np.savez(out_path + "_" + str(chunk), **res)
    print("FINISHED")
    print(elapsed, nb/elapsed)

class Hash:
    def __init__(self, ds, resizer=None):
        self.ds = ds
        self.resizer = resizer

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        # if self.transform is not None:
            # pass
        vals = []
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
                value = compute_hash(image_bytes)
                value_int = value.get_hash()
            except Exception as ex:
                print(ex)
                value = ''
                value_int = -1
                value_str = ""
            else:
                # value_str = f"{value_int:x}"
                value_str = str(value_int)
            vals.append(value_str)
        return (idx,) + tuple(vals)

    def __len__(self):
        return len(self.ds)

resizers = {
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
def downstream(*, dataset_name="cifar10", root="root", split="test", out_path="downstream", batch_size=10_000, workers=64, resizer=""):
    """
    Compute downstream (CLIP_benchmark) based hashes
    """
    name = dataset_name.replace("/", "-")
    out_path = f"{out_path}/{name}_{split}"
    from clip_benchmark.datasets.builder import (
        build_dataset,
        get_dataset_collate_fn,
        get_zeroshot_classification_templates,
    )
    from functools import partial
    import torch
    resizer = resizers[resizer] if resizer else None
    print(resizer)
    ds = Hash(build_dataset(
        dataset_name=dataset_name,
        root=root,
        download=True,
        split=split,
        transform=None,
        annotation_file="",
    ), resizer=resizer)
    print(len(ds))
    # collate_fn = get_dataset_collate_fn(dataset_name)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    hashes_all = defaultdict(list)
    step = 0
    for hashes in dl:
        inds, *hashes = hashes
        for i, name in enumerate(hash_names):
            for suffix in ("_str",):
                hashes_all[name+suffix].extend(hashes[i])
        print(step)
        step += 1
        print(inds)
    print("FINISHED")
    np.savez(out_path, **hashes_all)
    with open(out_path+".csv", "w") as fd:
        data = "\n".join(hashes_all["dct_hash_image_from_buffer_str"])
        fd.write(data)

def cross(out_path, src, *dsts):
    """
    Compute hashes that are both in upstream and downstream datasets (intersection)
    """
    path_src = src
    print("read src")
    src = ([l[:-1] for l in open(src).readlines()])
    print("dedup src")
    src_dedup = set(src)
    global_rows = []
    for dst in dsts:
        path_dst = dst
        print("read dst", dst)
        dst = ([l[:-1] for l in open(dst).readlines()])
        print("dedup dst")
        dst_dedup_dict = defaultdict(list)
        for i, d in enumerate(dst):
            dst_dedup_dict[d].append(i)
        dst_dedup = set(dst)
        print("src before dedup",len(src), "src after dedup", len(src_dedup))
        print("dst before dedup", len(dst), "dst after dedup", len(dst_dedup))
        I = src_dedup.intersection(dst_dedup)
        print("intersection",len(I))
        I = list(I)
        rows = []
        for h in I:
            for ind in dst_dedup_dict[h]:
                rows.append({"hash": h, "index": ind})
        global_rows.append({
            "path_src": path_src, "path_dst": path_dst, "nsrc": len(src), 
            "nsrc_dedup": len(src_dedup), "ndst": len(dst), "ndst_dedup": len(dst_dedup), "src_dst_inter_unique": len(I),
            "dups":len(rows)
        })
        df = pd.DataFrame(rows)
        df.to_csv(f"{out_path}/"+os.path.basename(path_dst), index=False)
    df = pd.DataFrame(global_rows)
    df.to_csv(f"{out_path}/dedup.csv", index=False)

def build_upstream_index(upstream="laion400m"):
    """
    Create an upstream index for easily extracting the urls corresponding to a hash 
    """
    import joblib
    cached = f"upstream/{upstream}/index.pkl"
    print(f"Creating an index on the upstream dataset {upstream}")
    src = pd.read_csv(f"upstream/{upstream}/result.csv", dtype={"phash": int, "url": str})
    # src = src.drop_duplicates(subset=["phash"])
    src.set_index("phash", inplace=True)
    src.sort_index(inplace=True)
    joblib.dump(src, cached)

def show_dups(*, upstream="laion400m", dedup_path="dedup_downstream", dataset_name="imagenet1k", out="out.html"):
    """
    Create an HTML static page with all dups, for visualization
    """
    import joblib
    import PIL
    from subprocess import call
    from clip_benchmark.datasets.builder import (
	build_dataset,
	get_dataset_collate_fn,
	get_zeroshot_classification_templates,
    )
    import hashlib
    cached = f"upstream/{upstream}/index.pkl"
    src = joblib.load(cached)
    assert os.path.exists(cached)
    root = "clip_benchmark_datasets/" + dataset_name
    split = "test"
    ds = build_dataset(
        dataset_name=dataset_name,
        root=root,
        download=True,
        split=split,
        transform=None,
        annotation_file="",
    )
    dst = pd.read_csv(f"{dedup_path}/{dataset_name}_test.csv", dtype={"hash": int, "index": int}).rename({"index": "ind"},axis=1).set_index("hash")
    # loop over unique hashes from downtream that exist on upstream
    N = 0
    lines = []
    invalid = set()
    # for val in dst.index.unique():
        # # get indices of the images from downtream that correspond to current hash
        # inds = dst.loc[val].ind
        # if type(inds) == np.int64:
            # inds = [inds]
        # else:
            # inds = inds.values
        # # loop over indices of images from downstream that correspond to current hash
        # for ind in inds:
            # #print(ind)
            # img, label = ds[ind] # get the image from downstream
            
            # # Get urls from upstream metadata corresponding to current hash
            # t0 = time.time()
            # di = src.loc[val]
            # print(time.time() - t0)
            # if len(di) == 1:
                # urls = [di.url]
            # else:
                # urls = di.url.values # get the url from upstream
            # # Display downstream image followed by upstream images with same hash, in a row
            # img = (img.resize((224, 224)))
            # by = BytesIO()
            # img.save(by, format='jpeg')
            # imgdata = by.getvalue()
            # # img = widgets.Image(value=imgdata)
            # #l  = [img]        
            # l = [to_img_tag(imgdata)]
            # urls = list(sorted(list(set(urls))))
            # urls = urls[:5]
            # for url in urls:
                # #print(url)
                # #html = f"<img src='{url}' width=224 height=224/>"
                # # if "1819teen" in url:
                    # # continue
                # uid = hashlib.md5(url.encode()).hexdigest()
                # path = f"cache/{uid}.jpg"
                # if path in invalid:
                    # continue
                # if not os.path.exists(path):
                    # call(f"wget '{url}' --output-document={path} --timeout=10 --tries=5", shell=True)
                # N += 1
                # try:
                    # img = PIL.Image.open(path).convert('RGB').resize((224,224))
                # except Exception as ex:
                    # print(ex)
                    # invalid.add(path)
                    # continue
                # by = BytesIO()
                # img.save(by, format='jpeg')
                # imgdata = by.getvalue()                
                # # wd = widgets.Image(value=imgdata)
                # # l.append(wd)
                # l.append(to_img_tag(imgdata)+f"{ind}")
                # print(val, url)
            # # display(HBox(l))
            # if len(l) > 1:
                # lines.append(l)

    dst = pd.read_csv(f"{dedup_path}/{dataset_name}_test.csv", dtype={"hash": str}).rename({"index": "ind"},axis=1).set_index("hash")
    dst = dst.reset_index().sort_values(by="ind")
    dst["ind"] = dst.ind.astype(str)
    print(dst.hash.dtype)
    print(src.index.dtype)
    print(dst.hash)
    for _, di in dst.iterrows():
        # di = dst.iloc[i]
        # di = dst[dst.ind==di]
        # print(di.hash, di.ind)
        si = src.loc[np.uint64(di.hash)]
        if len(si) == 1:
            urls = [si.url]
        else:
            urls = si.url.values # get the url from upstream
        print(di.ind)
        ind = int(di.ind)
        img, label = ds[ind] # get the image from downstream
        img = (img.resize((224, 224)))
        by = BytesIO()
        img.save(by, format='jpeg')
        imgdata = by.getvalue()
        l = [to_img_tag(imgdata)]
        urls = list(sorted(list(set(urls))))
        urls = urls[:5]
        for url in urls:
            uid = hashlib.md5(url.encode()).hexdigest()
            path = f"cache/{uid}.jpg"
            if path in invalid:
                continue
            if not os.path.exists(path):
                call(f"wget '{url}' --output-document={path} --timeout=10 --tries=5", shell=True)
            N += 1
            try:
                img = PIL.Image.open(path).convert('RGB').resize((224,224))
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
    with open(out, "w") as fd:
        fd.write("<html><table>" + table + f"</table><p>Total: {total}</p><html>")

def to_img_tag(data):
    import base64
    data_base64 = base64.b64encode(data)
    data_base64 = data_base64.decode()
    html = '<img src="data:image/jpeg;base64,' + data_base64 + '">'
    return html

def show_dup_filenames(*, dedup_path="dedup_downstream", dataset_name="imagenet1k", out="out.txt"):
    import joblib
    import PIL
    from subprocess import call
    from clip_benchmark.datasets.builder import (
	build_dataset,
	get_dataset_collate_fn,
	get_zeroshot_classification_templates,
    )
    root = "clip_benchmark_datasets/" + dataset_name
    split = "test"
    ds = build_dataset(
        dataset_name=dataset_name,
        root=root,
        download=True,
        split=split,
        transform=None,
        annotation_file="",
    )
    dst = pd.read_csv(f"{dedup_path}/{dataset_name}_test.csv", dtype={"hash": int, "index": int}).rename({"index": "ind"},axis=1)
    rows = []
    for ind in sorted(dst.ind.values):
        path, _ = ds.samples[ind]
        path = path.replace(root, "")[1:]
        print(path)
        rows.append(path)
    fd = open(out, "w")
    fd.write("\n".join(rows))
    fd.close()


if __name__ == "__main__":
    run([upstream, downstream, cross, build_upstream_index, show_dups, show_dup_filenames])
