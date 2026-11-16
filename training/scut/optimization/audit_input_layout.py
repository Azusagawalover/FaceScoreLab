"""Read-only diagnostic of historical channels-last versus current NCHW inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from runner import atomic_json, load_development_rows, setup_threads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dev-manifest", type=Path, required=True)
    p.add_argument("--new-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    setup_threads(4)
    rows = load_development_rows(args.dev_manifest)[:32]
    arrays = []
    for row in rows:
        with Image.open(args.data_dir/"crops"/(row["name"]+".png")) as opened:
            array = np.asarray(opened.convert("RGB"), dtype=np.float32)/255
        array = (array-np.array([.485,.456,.406], np.float32))/np.array([.229,.224,.225], np.float32)
        arrays.append(array.transpose(2,0,1))
    legacy_array = np.stack(arrays)
    current_array = np.stack([a.copy() for a in arrays])
    torch.manual_seed(42)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    prefix = nn.Sequential(*list(model.features.children())[:9]).eval()
    with torch.inference_mode():
        legacy_output = prefix(torch.from_numpy(legacy_array)).numpy()
        current_output = prefix(torch.from_numpy(current_array)).numpy()
    names = json.loads((args.data_dir/"frozen_early_names.json").read_text())
    lookup = {name:i for i,name in enumerate(names)}
    old = np.load(args.data_dir/"frozen_early_maps.npy", mmap_mode="r", allow_pickle=False)
    new = np.load(args.new_cache, mmap_mode="r", allow_pickle=False)
    old_values = np.asarray(old[[lookup[r["name"]] for r in rows]])
    new_values = np.asarray(new[0,:len(rows)]).copy()
    old._mmap.close()
    new._mmap.close()
    def difference(a,b):
        d = np.abs(a.astype(np.float64)-b.astype(np.float64))
        return {"max":float(d.max()),"mean":float(d.mean()),"different_values":int(np.count_nonzero(d))}
    result = {"n":len(rows),"test_targets_opened":False,"test_feature_rows_compared":0,
              "legacy_numpy_strides":list(legacy_array.strides), "current_numpy_strides":list(current_array.strides),
              "legacy_torch_channels_last":torch.from_numpy(legacy_array).is_contiguous(memory_format=torch.channels_last),
              "current_torch_contiguous":torch.from_numpy(current_array).is_contiguous(),
              "input_value_difference":difference(legacy_array,current_array),
              "legacy_layout_vs_historical_cache":difference(legacy_output,old_values),
              "current_layout_vs_historical_cache":difference(current_output,old_values),
              "current_layout_vs_new_cache":difference(current_output,new_values),
              "layout_output_difference":difference(legacy_output,current_output)}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
