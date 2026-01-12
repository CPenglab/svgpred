import os
from tqdm import tqdm
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchnet.meter import AverageValueMeter
from multiprocessing import Pool

from . import dataset, my_model

def train(dataset: Dataset,
          model,
          epochs,
          batch_size = 64,
          lr = 0.001,
          load_process = 0,
          save_dir = None,
          save_stride = 10,
          optim = None):

    ds = DataLoader(dataset,
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = load_process)

    # choose device
    device_finder = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_loader = torch.device(device_finder)
    
    model.to(device_loader)
    model.train()
    
    # loss & optimizer
    criterion = nn.CrossEntropyLoss().to(device_loader)
    if optim: 
        optimizer = optim(model.parameters()) 
    else:
        optimizer = Adam(model.parameters(), lr = lr)

    loss_meter = AverageValueMeter()

    # timer
    date = datetime.now()

    loss_name = f"avgLoss_{date.month}m{date.day}d" +\
                f"_{date.hour}h{date.minute}m{date.second}s.txt"


    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        loss_path = os.path.join(save_dir, loss_name)
    else:
        loss_path = os.path.join(".", loss_name)

    for epoch in range(epochs):
        print(f"{'-'*20}run epoch {epoch + 1}")

        loss_meter.reset()
        for ii, (name, data, label) in enumerate(tqdm(ds, ncols = 80)):
            data = Variable(data).to(device_loader)
            label = Variable(label).to(device_loader)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.cpu().detach())

        avg_loss = loss_meter.value()[0]
        print(f"epoch:{epoch + 1}; avg_loss:{avg_loss}")

        with open(loss_path, "a") as f:
            f.write(f"{float(avg_loss)},\n")

        save_name = f"model_{date.month}m{date.day}d_{date.hour}h" +\
                    f"{date.minute}m{date.second}s_epoch{epoch + 1}.pth"
        model_path = os.path.join(save_dir, save_name)

        if save_dir is not None and save_stride != 0:
            if (epoch + 1) % save_stride == 0:
                torch.save(model.state_dict(), model_path)
    
    if save_dir is not None:
        torch.save(model.state_dict(), model_path)
    
    return model

def tuning(num: int,
           data_root: str,
           ds_names: list,
           savedir: str):
    """
    Fine-tuning.

    ==========
    num: int 
        The index of the model to be trained.

    """

    residue = len(ds_names) - 1
    ds = dataset.ManualDS(
            comp_ds_num=num // residue,
            data_root=data_root,
            ds_names=ds_names,
            random_state=42)        
    ds.change_mode(mode="train", leave=num % residue)

    model = my_model.Densenet(pretrained=True)
    model.train()

    savepath = os.path.join(savedir, f"comp{num // residue}_{num % residue}")
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    train(ds, model, epochs = 3, save_dir = savepath, save_stride = 0,
            load_process=0)

def run_model(data_root: str,
              ds_names: list,
              savedir: str,
              proc = 10):
    func = partial(tuning,
                   data_root=data_root,
                   ds_names=ds_names,
                   savedir=savedir)
    
    # The number of models to be trained.
    num = len(ds_names) * (len(ds_names)-1)
    with Pool(proc) as pool:
        pool.map(func, range(num))



