import os
import numpy as np
import pandas as pd
import xarray as xr


def calculate_freq(ds, freq='5min', buffer='1H'):
    '''
    Calculate frequency given tip dataset.
    
    Parameters
    ----------
    ds: xarray.Dataset object containing one station's worth of tip data
    freq: time frequency string as in pandas - default '5min'
    buffer: time frequency string by which to increase max allowable time between
            tip reports - default '1H'

    Returns
    -------
    ds_freq: xarray.Dataset object with calculated accumulated rain at regular
             interals
    '''
    ds_freq = ds.resample(freq, 'time', how='sum', 
                          label='right', keep_attrs=True).fillna(0)

    buffer = pd.Timedelta(buffer)
    inter_tip_time =  pd.Timedelta(ds.rain_gage.attrs.get('inter_tip_time', 0))

    bool_array = ds.time.diff('time') > (inter_tip_time+buffer).asm8
    nan_times = np.stack([ds.time[:-1][bool_array], ds.time[1:][bool_array]]).T
    df_freq = ds_freq.rain_gage.to_series()

    for start_nan_time, end_nan_time in nan_times:
        df_freq[start_nan_time: end_nan_time] = np.nan

    ds_freq.rain_gage.values = np.array([df_freq.values]).T
    
    # keep those coords!
    ds_freq = ds_freq.assign_coords(lat=ds['lat'])

    return ds_freq


def freq_from_tips(path, files, freq='5min', buffer='1H'):
    '''
    Calculate frequency from list of tip files in path
    
    Parameters
    ----------
    path: path to directory in which tip files are located
    files: list of tip filenames (netcdf)
    freq: time frequency string as in pandas - default '5min'
    buffer: time frequency string by which to increase max allowable time between
            tip reports - default '1H'
    '''
    ds_list = []
    for file in files:
        ds = xr.open_dataset(os.path.join(path, file))
        ds_freq = calculate_freq(ds, freq, buffer)
        ds_list.append(ds_freq)

    ds_freq = xr.concat(ds_list, 'station')
    ds_freq.rain_gage.attrs = ds.rain_gage.attrs
    ds_freq.rain_gage.attrs.update({'calc_from_tips': True,
                                    'label': 'right',
                                    'freq': freq})
    return ds_freq