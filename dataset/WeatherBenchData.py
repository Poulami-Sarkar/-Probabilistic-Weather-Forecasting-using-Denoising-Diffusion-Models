# File imports
import zarr
import json
import os
import shutil
from datetime import datetime,timedelta
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import xarray as xr


from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, BatchSampler,SequentialSampler
from pathlib import Path

from config.config import *

class WeatherBenchData(Dataset):
    def __init__(self, data_dir: Path, split: str, batchsize: int = 8, subset: int = 1979,transforms: tuple = (0,1,0,1)):
        self.data_dir = data_dir
        self.prediction_distance =72
        self.transforms = transforms

        self.batchsize = batchsize
        self.channels = [x for x in self.data_dir.iterdir() if "constants" not in str(x)]
        self.channel_list = []

        # Add the list of channels to be added
        for i in self.channels [:]:
            files = [x for x in i.iterdir()]
            temp = []
            for f in files:
                f = xr.open_mfdataset([str(f)],engine="zarr")
                temp.append(f)
            self.channel_list.append(xr.combine_by_coords(temp,combine_attrs="override"))
            
        
        print("Total no of channels:",len(self.channel_list))

        # Join inputs
        self.inputs = xr.merge(self.channel_list, join="inner")
        self.inputs_orig = xr.merge(self.channel_list, join="inner")
        
        # Fetch target field : geopotential 500
        self.targets = self.inputs["z"].sel(level = 500)

        # Calculate mean and variances of the data:
        #self.get_mean_std("mean")
        """self.inputs_mean = self.inputs.mean(dim=["time","lon"])
        self.inputs_std = self.inputs.std(dim=["time","lon"])
        self.target_mean  = self.targets.mean(dim=["time","lon"])
        self.target_std = self.targets.std(dim=["time","lon"])"""

        print(self.inputs)
        # Train/test split
        if split == "train":
            self.inputs = self.inputs.isel(time = (self.inputs.time.dt.year.isin([range(subset,2016)])))
            self.targets = self.targets.isel(time=(self.targets.time.dt.year.isin([range(subset,2016)])))
        elif split == "val":
            self.inputs = self.inputs.isel(time=(self.inputs.time.dt.year == 2016))
            self.targets = self.targets.isel(time=(self.targets.time.dt.year == 2016))
        elif split == "test":
            self.inputs = self.inputs.sel(time=(self.inputs.time.dt.year >= 2017))
            self.targets = self.targets.sel(time=(self.targets.time.dt.year >= 2017))
        else:
            raise Exception("Incorrect split! (train or test)")

        self.mean = np.load("/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/dataset/mean.npy")
        self.std = np.load("/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/dataset/std.npy")
        self.length = self.inputs.dims['time']     
    

    def get_mean_std(self,type="mean"):
        
        # Flatten 13 levels into Channels
        l_3 =  []
        l_2 = []
        for i in list(self.inputs.data_vars):
            v = self.inputs[i].mean(dim=["time","lon"]).values  
            print(len(v.shape))
            if len(v.shape) == 1:
                l_3.append(v)
            else:
                l_2.append(v)
        self.stacked_3 = np.stack(l_3,axis=1)
        self.stacked_2 = np.vstack(l_2)
        input  = np.concatenate((self.stacked_2, self.stacked_3.T), axis=0)
        np.save("mean.npy", input)
        print(input.shape)


    def __getitem__(self,indices):
        
        
        # Convert indices to Numpy array        
        indices = np.array(indices)
        # Select batch
        input_fields = self.inputs.isel(time=indices)
        target_fields = self.targets.isel(time=indices+self.prediction_distance)
        
        # Flatten 13 levels into Channels
        l_3 =  []
        l_2 = []
        data_vars = ['v','t2m', 'v10', 'u', 't', 'vo', 'tisr', 'q', 'r', 'u10', 'tp', 'tcc', 'z', 'pv']
        for i in list(data_vars):
            v = input_fields[i].values  
            if len(v.shape) == 3:
                l_3.append(v)
            else:
                l_2.append(v)
        stacked_3 = np.stack(l_3,axis=1)
        stacked_2 = np.hstack(l_2)
        input_t = input_fields.time.values
        target_t = target_fields.time.values

        input  = np.concatenate((stacked_2, stacked_3), axis=1)
        input_  = np.concatenate((stacked_2, stacked_3), axis=1)
        
        targets_ = target_fields.values
        targets = target_fields.values
        
        if self.transforms:
            #print("Norm")
            input = (input - self.mean[: , : , np.newaxis]) / self.std[: , : , np.newaxis]
            mean = np.array([48819.805, 48833.81 , 48977.945, 49181.32 , 49708.07 , 50707.688,
                52079.074, 53543.01 , 54841.117, 55846.387, 56586.57 , 57109.504,
                57409.51 , 57496.883, 57479.42 , 57453.836, 57454.336, 57485.37 ,
                57522.215, 57486.332, 57265.863, 56813.133, 56153.152, 55324.438,
                54445.89 , 53666.86 , 53034.58 , 52537.387, 52102.508, 51710.734,
                51403.043, 51224.12 ])
            std = np.array([1288.7034 , 1363.7769 , 1403.0981 , 1434.1488 , 1517.5115 ,
                1657.6788 , 1680.4523 , 1536.7247 , 1303.5171 , 1054.1064 ,
                782.69867,  497.14798,  277.06384,  184.77434,  155.94571,
                147.91891,  148.33806,  160.48926,  207.96834,  320.28506,
                556.15424,  922.26685, 1326.1085 , 1664.5576 , 1869.0767 ,
                1947.1503 , 1980.6323 , 1995.312  , 1955.4883 , 1901.9044 ,
                1850.6393 , 1814.2471 ])
            #targets_ = (targets - 54115.75 ) / 3354.9412
            targets = (targets - mean[:,np.newaxis]) / std[:,np.newaxis]
            
            #(target_fields.values - self.mean[85 , : , np.newaxis]) / self.std[85 , : , np.newaxis]

        return {"day": (input_t - input_t.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) + 1,
                "hour":(input_t).astype('datetime64[h]').astype(int) % 24,
                "data":torch.tensor(input),
                "data_":torch.tensor(input_)}, {"day":(target_t - target_t.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) + 1,
                                              "hour":(target_t).astype('datetime64[h]').astype(int) % 24,
                                              "data":torch.tensor(targets,dtype=torch.float),
                                              "data_":torch.tensor(targets_,dtype=torch.float),

                                              }
    
    def __len__(self,):
        return  self.length-self.prediction_distance

    def standardize(self,transforms):
        
        inputs_mean,inputs_std,target_mean,target_std = transforms
        self.inputs = (self.inputs - inputs_mean)  / inputs_std
        self.targets = (self.targets - target_mean) / target_std
        

        pass

class WeatherBenchIterable(IterableDataset):
    def __init__(self,):
        pass

    def __iter__(self,):
        pass

def main():
    base_path =  "/home/sarkar/Documents/WeatherBench_12_unnorm" #
    t0 = datetime.now()
    weatherbench = WeatherBenchData(Path(base_path),"train")
    """
    inputs_mean = weatherbench.inputs.mean(dim=["time","lon"])
    inputs_std = weatherbench.inputs.std(dim=["time","lon"])
    target_mean  = weatherbench.targets.mean(dim=["time","lon"])
    target_std = weatherbench.targets.std(dim=["time","lon"])
    
    inputs_mean.to_zarr("inputs_mea.zarr", mode="w")
    inputs_std.to_zarr("inputs_std.zarr", mode="w")
    target_mean.to_zarr("target_mean.zarr", mode="w")
    target_std.to_zarr("target_std.zarr", mode="w") 
    
    inputs_mean = xr.open_mfdataset(["inputs_mean.zarr"],engine="zarr")
    inputs_std = xr.open_mfdataset(["inputs_std.zarr"],engine="zarr")
    target_mean  = xr.open_mfdataset(["target_mean.zarr"],engine="zarr")
    target_std = xr.open_mfdataset(["target_std.zarr"],engine="zarr")"""
    

    #weatherbench = WeatherBenchData(Path(base_path),"train")

    t1 = datetime.now()
    
    print("Length of datweatherbenchaset", len(weatherbench))


    print("Time to initialize dataset ", t1-t0)

    # Initialize dataloaders

    batch_size =32
    bs  = BatchSampler(SequentialSampler(weatherbench), batch_size=batch_size, drop_last=False)

    t3 = datetime.now()
    train_dataloader = DataLoader(weatherbench, sampler=BatchSampler(
        SequentialSampler(weatherbench), batch_size=batch_size, drop_last=False
    ))
    t4  =datetime.now()
    print("Time to init dataloader with batchsize ",batch_size, t4-t3)

    it = iter(train_dataloader)
    print("define iterator")
    t5_1 = datetime.now()
    first = next(it)
    t6_1 = datetime.now()
    print("Iternation 1", t6_1-t5_1)
    
    t5_2 = datetime.now()
    second = next(it)
    t6_2 = datetime.now()
    print("Iternation 2", t6_2 - t5_2)

    print(first[0]["data"].shape, first[1]["data"].shape)


if __name__ == '__main__':
    main()