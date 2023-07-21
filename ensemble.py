import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from . import dataset, my_model

def predict_(dataset: Dataset,
             model,
             batch_size = 50,
             proc = 0) -> pd.DataFrame:
    """
    Handling a binary classification problem using the input dataset and model.
    =================================
    dataset: torch.utils.data.Dataset
    model: Loaded model.
    proc: Number of processes used for loading the dataset.
    """
    pos = neg = 0

    info_names = []
    info_pos_svg = []
    info_neg_svg = []
    info_pred = []

    ds = DataLoader(dataset, batch_size = batch_size,
                            num_workers = proc)

    model.eval() # 以防万一
    with torch.no_grad():
        for ii, (name, data) in enumerate(tqdm(ds, ncols=80)):
            res = model(data)
            out = res.argmax(axis = 1)
            for n in range(len(out)):
                if out[n] == 0: 
                    neg += 1
                else:
                    pos += 1
                
                info_names.append(name[n])
                info_pos_svg.append(float(res[n, 1]))
                info_neg_svg.append(float(res[n, 0]))
                info_pred.append(int(out[n]))

    print(f"pos_num:{pos}, neg_num:{neg}\n{'-'*9}")

    svg = pd.DataFrame({"name": info_names, "svg_pos": info_pos_svg,
                        "svg_neg": info_neg_svg, "pred": info_pred})

    return svg

def pred(imgs_path: str, model_path: str, proc = 0):
    """
    prediction of SVGs.
    =============================
    imgs_path: str, The directory of gene imgs that generated for a dataset by draw.py
    model_path: str, The directory of the '.pth' file which will be used.    
    """

    ds = dataset.Pred_DS(dir_path = imgs_path)

    model = my_model.Resnet(pretrained = False)
    print(f"loading model {os.path.dirname(os.path.basename(model_path))}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    res = predict_(ds, model, proc=proc)
    return res

def mean_ensem_(data:list) -> pd.DataFrame:
    res = 0
    for d in data:
        res += d[["svg_neg", "svg_pos"]]
    res /= len(data)
    res.index = data[0]["name"].values
    res["pred"] = res.apply(np.argmax, axis = 1)
    
    return res

def mean_ensem(imgs_path:str, model_dir = "./models",
               proc = 10) -> pd.DataFrame:
    """
    Ensemble output.
    =================================
    imgs_path: str, The directory of gene imgs that generated for a dataset by draw.py
    model_dir: str, The directory containing all folders with '.pth' files.
    """

    model_list = [os.path.join(root, *sub, *file)
                  for root, sub, file in os.walk(model_dir)][1:]

    res = []
    for i in model_list:
        res.append(pred(i, imgs_path = imgs_path, proc=proc))

    me_res = mean_ensem_(res)
    me_res = me_res.loc[me_res["pred"] == 1, :]
    me_res = me_res.sort_values(by = "svg_pos", ascending = False)

    return me_res



