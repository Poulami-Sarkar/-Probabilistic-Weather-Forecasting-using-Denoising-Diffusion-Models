# Import libraries
import matplotlib.pyplot as plt
import climetlab as cml
from pathlib import Path
import os
import xarray as xr
import numpy as np
import dask

# TODO: Add converter for constants

def get_mean_std(data):
    #channels = [x for x in base_path.iterdir() if "constants" not in str(x)]
    channel_list = []
    #print(channels)
    for i in data:
        print(i)
        channel_list.append(xr.open_mfdataset(str(i)+"/"+"*.nc",combine="by_coords"))
    inputs = xr.merge(channel_list, join="inner")
    mean = inputs.mean(dim=["time","lon"])
    std = inputs.std(dim=["time","lon"])

    return mean, std


def sanity_check(data):
    """ Sanity check the dataset (Only the number of files)

    Args:
        data (array): Path to data directories

    Raises:
        Exception: if no of files does not match

    Returns:
        Boolean: Returns True if sanity check passes
    """
    # return True if len(data) == 17 else raise Exception("Sanity check failed. No of files should be 17") 
    for i in data[:]:
        if  len(os.listdir(i)) != 40:
            raise Exception("Sanity check failed. No of files should be 40") 
            
    return True

def convert_constants(base_path, write_path, chunk):
    raise NotImplementedError

def convert_to_zarr(base_path, write_path,chunk, norm = True):
    """Concatenate data files

    Args:
        base_path (String): Path to data directory

    Returns:
        dict: dictionary of data fields
    """
    
    base_path = Path(base_path)
    
    
    data = [x for x in base_path.iterdir() if x.is_dir()]
    sanity_check(data)
    d = []
    #data_mean,data_std= get_mean_std(data)
    #print(data_mean, data_std)
    
    # Extract files
    # 0,1,2,6,9,11

    for i in data[7:8]:
        files = [x for x in i.iterdir()]
        dir_path = str(i).split("/")[-1]
        if not os.path.exists(write_path+dir_path):
            print("Creating directory",dir_path)
            os.mkdir(write_path+dir_path)

        data_mean,data_std= get_mean_std([str(i)])
        f = xr.open_mfdataset([str(files[0])])
        var_name = list(f.data_vars.keys())
        mean = data_mean[var_name[0]]
        std = data_std[var_name[0]]
        print(data_mean)
        print(data_std)

        if norm:
            print("Normalizing data")

        for f in files:
            file_path = str(f).split("/")[-1].replace(".nc",".zarr")
            f = xr.open_mfdataset([str(f)])
            #print(dir_path)
            #print(f.dims)
            #break
            chunk_size = dict(f.dims)
            chunk_size['time'] = chunk
            
            if norm:
                if len(var_name) != 1:
                    print("File format error")
                    exit()
                
                #OLD METHOD
                """if "level" not in f.coords: 
                    #f = f.expand_dims({'level': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]})
                    #f = f.assign_coords(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
                    #f = f.transpose('time', 'level', 'lat', 'lon')

                    # Calculate the mean and standard deviation of the variable along the time and lat dimensions
                    mean = data_mean[""]
                    std = 
                else:
                    # Calculate the mean and standard deviation of the variable along the time and lat dimensions
                    mean = f.mean(dim=['time', 'lat','lon','level'])
                    std = f.std(dim=['time', 'lat','lon','level'])
                """
                # Standardize the variable
                f = dask.array.from_array(f,chunks = chunk_size)
                mean = dask.array.from_array(mean)
                std = dask.array.from_array(std)

                f  = (f - mean) / std            

            # Write to disk
            save_path = write_path+dir_path+"/"+file_path
            print("Saving file to disk "+ save_path)
            chunk_size = dict(f.dims)
            chunk_size['time'] = chunk
            item = f.chunk(chunk_size)
            #with dask.config.set(scheduler='threads'): 
            item.to_zarr(save_path, mode="w")
        

def main():
    """Main function
    """

    # Set path of the data directory and extract the data
    base_path = "/media/sarkar/data/sarkar/weatherbench/"
    write_path = "/home/sarkar/Documents/WeatherBench_12_unnorm/"
    convert_to_zarr(base_path, write_path, 12,False)

if __name__ == '__main__':
    main()