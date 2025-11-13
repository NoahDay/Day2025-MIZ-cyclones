from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
from simple_maths import getClosestPoint1D, getClosestPoint2D
import os

def convertFilenameToDatetime(filename, model_timestep='h'):
    if model_timestep == 'h':
        date_time_str = filename.split('.')[1]
        date_str, seconds_str = date_time_str[:10], date_time_str[11:]
        date_part = datetime.strptime(date_str, "%Y-%m-%d")
        seconds_part = timedelta(seconds=int(seconds_str))
        output = date_part + seconds_part
    elif model_timestep == 'd':
        date_time_str = filename.split('.')[1]
        date_str, seconds_str = date_time_str[:10], date_time_str[11:]
        date_part = datetime.strptime(date_str, "%Y-%m-%d")
        output = date_part 
    return output


def getCycloneDatetime(df_single_track):
    '''
    Convert the times in df_single_track to pandas datetimes.
    '''
    timesteps,cols = df_single_track.shape
    dates = []
    for idx in range(timesteps):
        temp_date = datetime(df_single_track['year'].values[idx], df_single_track['month'].values[idx], df_single_track['day'].values[idx], 
                             df_single_track['hour'].values[idx])
        dates.append(temp_date)
        
    dates = pd.to_datetime(dates)
    return dates

def getLongtiudeIndices(df_single_track, ds_CICE, buffer, dataset="CICE"):
    '''
    Given the coordinate range from df_single_track, return the indicies which cover this subdomain in ds_CICE.
    '''
    min_lon, max_lon = df_single_track.longitude.min(), df_single_track.longitude.max()
    if abs(min_lon - max_lon) > 270:
        over_seam = True
        temp_lons = df_single_track.longitude.apply(lambda x: x - 360 if x > 270 else x)
        min_lon, max_lon = temp_lons.min(), temp_lons.max()
    else:
        over_seam = False
    _, min_lon_idx = getClosestPoint1D(np.array(df_single_track.longitude.values), min_lon)
    min_lon_lat = df_single_track.latitude.values[min_lon_idx]
    _, max_lon_idx = getClosestPoint1D(np.array(df_single_track.longitude.values), max_lon)
    max_lon_lat = df_single_track.latitude.values[max_lon_idx]
    
    if dataset == "CICE":
        if over_seam:
            target = np.array([min_lon+360, min_lon_lat])
        else:
            target = np.array([min_lon, min_lon_lat])
        closest_coord_min_lon, closest_2d_index_min_lon = getClosestPoint2D(ds_CICE.TLON.values, ds_CICE.TLAT.values, target)
        target = np.array([max_lon, max_lon_lat])
        closest_coord_max_lon, closest_2d_index_max_lon = getClosestPoint2D(ds_CICE.TLON.values, ds_CICE.TLAT.values, target)
        if abs(closest_2d_index_min_lon[1] - closest_2d_index_max_lon[1]) > 700:
            range_1 = np.arange(closest_2d_index_min_lon[1] - buffer, ds_CICE.ni.values.max(), 1)
            range_2 = np.arange(ds_CICE.ni.values.min(), closest_2d_index_max_lon[1] + buffer, 1)
            ni_range = np.concatenate((range_1, range_2))
        else:
            ni_range = np.arange(closest_2d_index_min_lon[1] - buffer, closest_2d_index_max_lon[1] + buffer, 1)
        ni_range

    elif dataset == "jra55":
        if over_seam:
            target = np.array([min_lon+360])
        else:
            target = np.array([min_lon])
        
        closest_coord_min_lon, closest_1d_index_min_lon = getClosestPoint1D(ds_CICE.lon.values, target)
        target = np.array([max_lon])
        closest_coord_max_lon, closest_1d_index_max_lon = getClosestPoint1D(ds_CICE.lon.values, target)
        if closest_1d_index_min_lon < closest_1d_index_max_lon:
            print('True')
            ni_range = np.arange(closest_1d_index_min_lon, closest_1d_index_max_lon)
        else:
            ni_range = np.arange(closest_1d_index_max_lon, closest_1d_index_min_lon)

    return ni_range

def fixCiceDatetimes(ds_in, model_timestep='h'):
    timestamps = ds_in.time.values
    dt_index = pd.DatetimeIndex(timestamps)
    # Subtract one hour
    if model_timestep == 'h':
        dt_index_minus_1h = dt_index - pd.Timedelta(hours=1)
        ds_out = ds_in.copy()
#    ds_out.time.values = dt_index_minus_1h.to_numpy()
        ds_out = ds_out.assign_coords(time=dt_index_minus_1h)
    elif model_timestep == 'd':
        dt_index_minus_12h = dt_index - pd.Timedelta(hours=12)
        ds_out = ds_in.copy()
        ds_out = ds_out.assign_coords(time=dt_index_minus_12h)
    return ds_out

def subdomainCiceAndJra55(df_single_track, ds_CICE, ds_jra55, buffer=30):
    ni_range = getLongtiudeIndices(df_single_track, ds_CICE, buffer)
    ds_sel = ds_CICE.sel(ni=ni_range)
    
    lon_min, lon_max = df_single_track.longitude.min(), df_single_track.longitude.max()
    if abs(lon_min - lon_max) > 270:
        over_seam = True
        temp_lons = df_single_track.longitude.apply(lambda x: x - 360 if x > 270 else x)
        lon_min, lon_max = temp_lons.min(), temp_lons.max()
        ds_sel_jra55 = xr.concat([ds_jra55.sel(lon=slice(lon_min + 360 - buffer, 360)), ds_jra55.sel(lon=slice(0, lon_max + buffer))], dim='lon')
    else:
        ds_sel_jra55 = ds_jra55.sel(lon=slice(lon_min, lon_max))
    
    return ds_sel, ds_sel_jra55


def changeInSIC(ds_CICE_in):
    ds_CICE_tmp = ds_CICE_in.copy()
    ds_CICE_tmp['daidt'] = ds_CICE_tmp['daidtd'] 
    ds_CICE_tmp['daidt'].values += ds_CICE_tmp['daidtt'].values
    ds_CICE_tmp['daidt'].attrs['long_name'] = ds_CICE_tmp['daidtd'].attrs['long_name'][:-8] + 'both'
    return ds_CICE_tmp


def fixCyclicPoint(data, datatype='jra'):
    if datatype=='jra':
        if abs(np.diff(data.longitude, 1)).max() > 180:
            temp_lons = data.longitude.apply(lambda x: x - 360 if x > 180 else x)
            data.longitude = temp_lons
        else:
            temp_lons = data.longitude
    elif datatype=='CICE_lon':
        if abs(abs(np.diff(data.lon.values,1))).max() > 180:
            temp_lons = data.lon.values
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data.lon.values
    elif datatype=='CICE':
        if abs(abs(np.diff(data.TLON.values,1))).max() > 180:
            temp_lons = data.TLON.values
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data.TLON.values
    else:
        if abs(abs(np.diff(data,1))).max() > 180:
            temp_lons = data
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data

    return temp_lons


# CODE TO PUT INTO FUNCTIONS
def makeCycloneTrackDataset(df_single_track, ds_CICE):
    df_single_track['date'] = pd.to_datetime(df_single_track['date'])

    # Create the DataArray
    xarray_tmp = xr.DataArray(
        data=df_single_track['central_pressure [hPa]'],  # Assuming latitude is already aligned properly
        dims=['time'],
        coords={
            'time': df_single_track['date']
        },
        name='central_pressure'
    )
    
    dataset = xarray_tmp.to_dataset(name='central_pressure')
    
    dataset['longitude'] = ('time', df_single_track['longitude'])
    dataset['latitude'] = ('time', df_single_track['latitude'])
    dataset['laplacian_of_central_pressure'] = ('time', df_single_track['laplacian_of_central_pressure [hPa/degree latitude**2]'])
    dataset['cyclone_depth'] = ('time', df_single_track['cyclone_depth [hPa]'])
    dataset['cyclone_radius'] = ('time', df_single_track['cyclone_radius [degrees latitude]'])
    dataset['eastward_steering_velocity'] = ('time', df_single_track['eastward_steering_velocity [m/s]'])
    dataset['northward_steering_velocity'] = ('time', df_single_track['northward_steering_velocity [m/s]'])
    
    dataset['longitude'].attrs['units'] = 'degrees'
    dataset['latitude'].attrs['units'] = 'degrees'
    dataset['laplacian_of_central_pressure'].attrs['units'] = 'hPa/degree latitude^2'
    dataset['cyclone_depth'].attrs['units'] = 'hPa'
    dataset['cyclone_radius'].attrs['units'] = 'degrees latitude'
    dataset['eastward_steering_velocity'].attrs['units'] = 'm/s'
    dataset['northward_steering_velocity'].attrs['units'] = 'm/s'
    
    ds_CICE['time'] = pd.to_datetime(ds_CICE.time)
    
    # Now perform the interpolation
    dataset_interp = dataset.interp(time=ds_CICE.time)
    dataset_interp
    return dataset_interp

def getCrossingTime(ds_CICE_in, xarrayTrack_in):
    deg_lat_between_cyclone_and_ice_edge = []
    times = []
    
    # Check if all values in ds_CICE_in['TLON'] are NaN
    if np.all(np.isnan(ds_CICE_in['TLON'].values)):
        print('All NaNs in CICE')
        return None, None, None
    
    for i, time in enumerate(xarrayTrack_in.time):
        ds_tmp = ds_CICE_in.sel(time=time)
        
        # Check for NaN in time
        if np.isnan(time):
            print(f"ERROR: NAN in time at index {i}")
            deg_lat_between_cyclone_and_ice_edge.append(np.inf)
            continue
        
        # Check for NaN in longitude
        longitude = xarrayTrack_in['longitude'].sel(time=time).values
        if np.isnan(longitude):
            print(f"Warning: NAN in longitude at index {i}")
            deg_lat_between_cyclone_and_ice_edge.append(np.inf)
            continue
        
        # Identify the ni index closest to the cyclone's longitude
        cyclone_ni = int(abs(ds_tmp['TLON'] - longitude).argmin(dim='ni').mean())
        
        # Identify the nj index corresponding to the ice edge
        ice_mask = ds_tmp['aice'] > 0.15
        ice_edge_nj = ds_tmp['TLAT'].where(ice_mask).sel(ni=cyclone_ni).argmax(dim='nj')
        
        # Get coordinates of the ice edge
        ice_edge_coord = [
            ds_tmp['TLON'].sel(ni=cyclone_ni, nj=ice_edge_nj).values,
            ds_tmp['TLAT'].sel(ni=cyclone_ni, nj=ice_edge_nj).values
        ]
        
        # Calculate the difference in latitude between the cyclone and the ice edge
        deg_lat_diff = (xarrayTrack_in['latitude'].sel(time=time) - ice_edge_coord[1]).values
        deg_lat_between_cyclone_and_ice_edge.append(deg_lat_diff)
        times.append(time.values)
    
    # Convert the list to a numpy array and find the minimum distance
    deg_lat_between_cyclone_and_ice_edge = np.array(deg_lat_between_cyclone_and_ice_edge).flatten()
    if abs(deg_lat_between_cyclone_and_ice_edge).min() > 1.0:
        print("Cyclone doesn't cross ice edge (> 1 deg.)")
        return None, None, None
        
    crossing_idx = abs(deg_lat_between_cyclone_and_ice_edge).argmin()
    crossing_time = times[crossing_idx]
    
    return crossing_time, crossing_idx, deg_lat_between_cyclone_and_ice_edge



def cyclic_slice(ds, dim, start, stop, step=1):
    # Get the size of the dimension
    dim_size = ds.sizes[dim]
    
    # Adjust the start index if it's negative
    if start < 0:
        start = dim_size + start
    
    # Apply the slice
    return ds.isel({dim: slice(start, stop, step)})


def getIceEdgeMask(ds_CICE, crossing_time, radius_length):
    """
    Get the mask at the ice edge over the cyclone crossing longitudes.

    Parameters:
    - ds_CICE: xarray.Dataset containing CICE outputs for the duration of the storm
    - crossing_time: datetime or index, the time at which the cyclone crosses the ice edge
    - radius_length: int, radius specified over which the mask should be calculated

    Returns:
    - ds_slice: xarray.Dataset, the sliced dataset around the cyclone's path
    - crossing_lons_mask_diameter: xarray.DataArray, mask at the ice edge across the cyclone's crossing longitudes
    - idx_ni_diameter: slice, ni indices that correspond with the crossing longitudes
    """
    # Select the cyclone's path at the specified crossing time
    crossing_edge_mask = ds_CICE['cyclone_path'].sel(time=crossing_time)
    cyclone_path_at_time = crossing_edge_mask.values
    
    # Find the index of the maximum value in the cyclone's path (assumed to be the center)
    idx = np.argmax(cyclone_path_at_time)
    idx_nj, idx_ni = np.unravel_index(idx, cyclone_path_at_time.shape)

    # Define the range for the neighborhood around the cyclone
    dx = 20
    dy = 20
    nj_range = slice(idx_nj - dy, idx_nj + dy + 1)

    # Apply cyclic slicing on the 'ni' dimension and then slice along the 'nj' dimension
    ds_slice = cyclic_slice(ds_CICE, 'ni', idx_ni - dx, idx_ni + dx + 1)
    ds_slice = ds_slice.isel(nj=nj_range)
    
    # Create a mask that will accumulate the masks over the specified radius
    crossing_edge_mask = ds_slice['cyclone_path'].sel(time=crossing_time)
    crossing_lons_mask = np.zeros_like(crossing_edge_mask)
    
    idx_ni_length = slice(idx_ni-radius_length,idx_ni+radius_length+1)

    # Accumulate the mask within the specified radius
    for i in np.arange(-radius_length, radius_length + 1):
        crossing_lons_mask += crossing_edge_mask.roll(ni=int(i))
    
    return ds_slice, crossing_lons_mask, idx_ni_length

def makeCycloneTrackDataset(df_single_track, ds_CICE):
    df_single_track['date'] = pd.to_datetime(df_single_track['date'])

    # Create the DataArray
    xarray_tmp = xr.DataArray(
        data=df_single_track['central_pressure [hPa]'],  # Assuming latitude is already aligned properly
        dims=['time'],
        coords={
            'time': df_single_track['date']
        },
        name='central_pressure'
    )
    
    dataset = xarray_tmp.to_dataset(name='central_pressure')
    
    dataset['longitude'] = ('time', df_single_track['longitude'])
    dataset['latitude'] = ('time', df_single_track['latitude'])
    dataset['laplacian_of_central_pressure'] = ('time', df_single_track['laplacian_of_central_pressure [hPa/degree latitude**2]'])
    dataset['cyclone_depth'] = ('time', df_single_track['cyclone_depth [hPa]'])
    dataset['cyclone_radius'] = ('time', df_single_track['cyclone_radius [degrees latitude]'])
    dataset['eastward_steering_velocity'] = ('time', df_single_track['eastward_steering_velocity [m/s]'])
    dataset['northward_steering_velocity'] = ('time', df_single_track['northward_steering_velocity [m/s]'])
    
    dataset['longitude'].attrs['units'] = 'degrees'
    dataset['latitude'].attrs['units'] = 'degrees'
    dataset['laplacian_of_central_pressure'].attrs['units'] = 'hPa/degree latitude^2'
    dataset['cyclone_depth'].attrs['units'] = 'hPa'
    dataset['cyclone_radius'].attrs['units'] = 'degrees latitude'
    dataset['eastward_steering_velocity'].attrs['units'] = 'm/s'
    dataset['northward_steering_velocity'].attrs['units'] = 'm/s'
    
    ds_CICE['time'] = pd.to_datetime(ds_CICE.time)
    
    # Now perform the interpolation
    dataset_interp = dataset.interp(time=ds_CICE.time)
    dataset_interp
    return dataset_interp


def getEachBreakupIdx(breakup_idx):
    '''
    Given an boolean array of when the breakup threshold has been met return each breakup event. 
    A breakup event is defined as when continued breakup occurs.

    Input: Boolean array
    Output: Stacked boolean array of each breakup event
    '''

    int_arr = breakup_idx.astype(int)
    # Add an artificial value from the start
    int_arr = np.insert(int_arr, 0, 0)
    diff = np.diff(int_arr)
    transition_indexes = np.where(diff == 1)[0]
    transition_indexes_end = np.where(diff == -1)[0] + 1
    
    # Add a starting point if the first element in True
    # if transition_indexes[0]:
    #     transition_indexes = np.insert(transition_indexes, 0, 0)
    
    breakup_idx_stacked = []
    
    # Stack these breakup events
    for i in np.arange(0, len(transition_indexes), 1):
        new_array = np.full(breakup_idx.shape, False)
        if i >= len(transition_indexes_end):
            new_array[transition_indexes[i]:] = breakup_idx[transition_indexes[i]:]
            # print( breakup_idx[transition_indexes[i]])
        else:
            new_array[transition_indexes[i]:transition_indexes_end[i]] = breakup_idx[transition_indexes[i]:transition_indexes_end[i]]
            # print(breakup_idx[transition_indexes[i]:transition_indexes_end[i]+1])
        breakup_idx_stacked.append(new_array)
    
    # Check to see if there are any errors
    if ~(np.sum(breakup_idx_stacked, axis=0).astype(bool) == breakup_idx).all():
        print("Error: Breakup indexes not retained!")
        print(np.sum(breakup_idx_stacked, axis=0).astype(bool))
        print(breakup_idx)

    return breakup_idx_stacked

def getMajorEvents(breakup_events_array, time_threshold=6):
    # Given a stacked array of all breakup events return those that persist for longer than time_threshold:
    idx_bool = np.sum(breakup_events_array, axis=1) >= time_threshold
    idx = np.where(idx_bool)[0]

    major_events_stacked = []
    for i in idx:
        major_events_stacked.append(breakup_events_array[i])
    return major_events_stacked

# def append_to_dataframe(dataframe, row_data):
#     new_row = pd.DataFrame([row_data], columns=dataframe.columns)
#     dataframe = pd.concat([dataframe, new_row], ignore_index=True)
#     return dataframe

# ice_edge_path = '/g/data/gv90/nd0349/antarctic-cyclones/' + 'ice-edge/'
# ice_edge_identifier = '_ice_edge.nc'
# directory_path = ice_edge_path + '*' + ice_edge_identifier
# files = sorted(glob.glob(directory_path))

# area_path = netcdfpath + 'area/'
# area_identifier = '_area.nc'
# # xr.open_dataset('/Volumes/NoahDay1TB/antarctic-cyclones/area/201519169_area.nc')['wave_sig_ht_interior'].isel(radius=2).plot()

# colvars = ['time', 'track', 'relative time', 'event',
#            'miz_width_init', 'miz_width_ave', 'miz_width_max', 'delta_miz_width', 'delta_miz_width_signed',
#             'ice_edge_init', 'ice_edge_ave', 'ice_edge_max', 'delta_ice_edge', 'delta_ice_edge_signed',
#             'wave_sig_ht_init', 'wave_sig_ht_ave', 'wave_sig_ht_max', 'delta_wave_sig_ht', 'delta_wave_sig_ht_signed',
#             'dafsd_wave_ra_init', 'dafsd_wave_ra_ave', 'dafsd_wave_ra_max', 'delta_dafsd_wave_ra', 'delta_dafsd_wave_ra_signed',
#             # 'wave_sig_ht_miz_init', 'wave_sig_ht_miz_ave', 'wave_sig_ht_miz_max', 'wave_sig_ht_miz_ra', 'wave_sig_ht_miz_signed',
#             # 'wave_sig_ht_interior_init', 'wave_sig_ht_interior_ave', 'wave_sig_ht_interior_max', 'delta_wave_sig_ht_interior', 'delta_wave_sig_ht_interior_signed',
#             # 'fsdrad_miz_init', 'fsdrad_miz_ave', 'fsdrad_miz_max', 'fsdrad_miz_ra', 'delta_fsdrad_miz_signed',
#             # 'fsdrad_interior_init', 'fsdrad_interior_ave', 'fsdrad_interior_max', 'fsdrad_interior_ra', 'delta_fsdrad_interior_signed',
#             # 'aice_miz_init', 'aice_miz_ave', 'aice_miz_max', 'aice_miz_ra', 'delta_aice_miz_signed',
#             # 'aice_interior_init', 'aice_interior_ave', 'aice_interior_max', 'aice_interior_ra', 'delta_aice_interior_signed',
# ]
#            #'delta_ice_edge', 'delta_miz_width', 'delta_miz_width_sum', 'delta_miz_width_signed','dist', 'psl', 'ice_edge', 'ice_edge_std', 'uvel', 'vvel', 'uatm', 'vatm', 
#            #'Tair', 'Tsfc', 'daicedt', 'frazil', 'congel', 'iage', 'sst', 'wave_sig_ht', 'dafsd_wave_ra', 'miz_width', 'miz_width_std']
# df_breakup = pd.DataFrame(columns=colvars)

# storm_proximity = []
# min_psl = []

# THRESHOLD_STORM_PROXIMITY = 600 # 600/110 # 600 km from Hepworth, converted to deg. latitude
# THRESHOLD_DAFSD_WAVE_RA = -10.0 # m/h WAS -20
# THRESHOLD_BREAKUP_TIME = 6 # Hours
# ROLLING_TIME = 1
# width = 2


# def addVarQuantities(ds, var, tmp_row, event_flag_tmp):
#     first_idx = np.argmax(event_flag_tmp) 
#     delta_var = ds[var].isel(radius=width).diff(dim='time')
#     tmp_row.append(ds[var].isel(radius=width).values[first_idx])
#     tmp_row.append(ds[var].isel(radius=width).values[event_flag_tmp].mean())
#     tmp_row.append(ds[var].isel(radius=width).values[event_flag_tmp].max())
#     tmp_row.append(abs(delta_var[event_flag_tmp[1:]]).sum().values)
#     tmp_row.append((delta_var[event_flag_tmp[1:]]).sum().values)
#     return tmp_row

# for file in tqdm(files[:]):
#     variables_to_drop = ['breakup_event', 'breakup_event_mask']
#     ds = xr.open_mfdataset(file, combine='by_coords', drop_variables=variables_to_drop)
#     track_number = file.split('/')[-1].split('_')[0]

#     # file_area = area_path + track_number + area_identifier
#     # if os.path.exists(file_area):
#     #     ds_area = xr.open_mfdataset(file_area, combine='by_coords', drop_variables=variables_to_drop)
#     # else:
#     #     #print(file_area)
#     #     continue

#     if np.all(np.isnan(ds['psl'].isel(radius=width).values)):
#         continue
#     min_psl_idx = ds['psl'].isel(radius=width).argmin().values
#     # Skip if the cyclone doesn't come close
#     if abs(ds['dist'].isel(radius=width)).min().values > THRESHOLD_STORM_PROXIMITY:
#         print('storm too far away')
#         continue

#     flg_cyclone_dist = ds['psl'].isel(radius=width).values < ds['psl'].isel(radius=width).mean().values #abs(ds['dist'].isel(radius=width).values) < THRESHOLD_STORM_PROXIMITY

#     delta_miz_width = ds['miz_width'].isel(radius=width).diff(dim='time')
#     delta_ice_edge = ds['ice_edge'].isel(radius=width).diff(dim='time')

#     # Breakup event
#     event = (ds['dafsd_wave_ra'].isel(radius=width)).rolling(time=ROLLING_TIME).mean()
#     event_idx = event < THRESHOLD_DAFSD_WAVE_RA
#     events_array = getEachBreakupIdx(event_idx)
#     if events_array:
#         # print('Breakup!')
#         major_events_array = getMajorEvents(events_array, time_threshold=THRESHOLD_BREAKUP_TIME)
#         if major_events_array:
#             event_flag = np.sum(major_events_array, axis=0) != 0
#             change_during_breakup = delta_miz_width[event_flag[1:]].sum().values
#             change_not_during_breakup = delta_miz_width[~event_flag[1:]].sum().values

#             for i_event in range(np.shape(major_events_array)[0]):
#                 event_flag_tmp = major_events_array[i_event]
#                 # colvars = ['time', 'event', 'relative time', 'delta_ice_edge', 'delta_miz_width',
#                 tmp_row = []
#                 tmp_row = [ds['time'][event_flag_tmp].mean().values]
#                 tmp_row.append(track_number)
#                 tmp_row.append(ds['time'].isel(time=ds['psl'].isel(radius=width).argmin().values).values - ds['time'][event_flag_tmp].mean().values)
#                 tmp_row.append('major_breakup')
# #                'miz_width_init', 'miz_width_ave', 'miz_width_max', 'delta_miz_width', 'delta_miz_width_signed',
                
#                 # Init var
#                 tmp_row = addVarQuantities(ds, 'miz_width', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'ice_edge', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'wave_sig_ht', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'dafsd_wave_ra', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'wave_sig_ht_miz', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'wave_sig_ht_interior', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'fsdrad_miz', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'fsdrad_interior', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'aice_miz', tmp_row, event_flag_tmp)
#                 # tmp_row = addVarQuantities(ds_area, 'aice_interior', tmp_row, event_flag_tmp)

#                 df_breakup = append_to_dataframe(df_breakup, tmp_row)


#     # Minor breakup event
#     event = (ds['dafsd_wave_ra'].isel(radius=width)).rolling(time=ROLLING_TIME).mean()
#     event_idx = event > THRESHOLD_DAFSD_WAVE_RA
#     events_array = getEachBreakupIdx(event_idx)
#     if events_array:
#         # print('Breakup!')
#         major_events_array = getMajorEvents(events_array, time_threshold=1)
#         if major_events_array:
#             event_flag = np.sum(major_events_array, axis=0) != 0
#             change_during_breakup = delta_miz_width[event_flag[1:]].sum().values
#             change_not_during_breakup = delta_miz_width[~event_flag[1:]].sum().values

#             for i_event in range(np.shape(major_events_array)[0]):
#                 event_flag_tmp = major_events_array[i_event]
#                 # colvars = ['time', 'event', 'relative time', 'delta_ice_edge', 'delta_miz_width',
#                 tmp_row = []
#                 tmp_row = [ds['time'][event_flag_tmp].mean().values]
#                 tmp_row.append(track_number)
#                 tmp_row.append(ds['time'].isel(time=ds['psl'].isel(radius=width).argmin().values).values - ds['time'][event_flag_tmp].mean().values)
#                 tmp_row.append('minor_breakup')
# #                'miz_width_init', 'miz_width_ave', 'miz_width_max', 'delta_miz_width', 'delta_miz_width_signed',
                
#                 # Init var
#                 tmp_row = addVarQuantities(ds, 'miz_width', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'ice_edge', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'wave_sig_ht', tmp_row, event_flag_tmp)
#                 tmp_row = addVarQuantities(ds, 'dafsd_wave_ra', tmp_row, event_flag_tmp)

#                 df_breakup = append_to_dataframe(df_breakup, tmp_row)
                
    

#     # del ds

# for var in colvars:
#     df_breakup[var] = df_breakup[var].apply(lambda x: x.item() if isinstance(x, np.ndarray) else x)

# # df_breakup = df_breakup.dropna()
# # df_breakup.to_csv(savepath+'csvs/predicting_breakup_n_{}.csv'.format(len(files)), index=False)
# df_breakup.head()
# df_breakup.to_csv(netcdfpath+'csvs/predicting_breakup_n_{}.csv'.format(len(files)), index=False)