import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import time
import logging
# import intake
# import requests
# import easygems.healpix as egh
# from functools import partial
import dask
from dask.distributed import Client, LocalCluster

#-------------------------------------------------------------------
def combine_masks(ds_mcs, ds_ar, client=None, out_zarr=None, logger=None):
    """
    Combine MCS and AR tracking datasets and write to Zarr store.
    
    Args:
        ds_mcs: xarray.Dataset
            MCS tracking dataset
        ds_ar: xarray.Dataset
            AR tracking dataset
        client: dask.distributed.Client, optional
            Dask client for distributed computation
        out_zarr: str, optional
            Output Zarr store path
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        xarray.Dataset: Combined dataset with all masks
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    drop_var_list = ['ccs_mask']
    rename_dict = {
        'AR_binary_tag': 'ar_mask',
        # 'TC_binary_tag': 'tc_mask',
    }

    # Check the calendar type of the time coordinate
    calendar = ds_ar['time'].dt.calendar

    # Convert DataSet time coordinate to standard calendar
    if calendar not in ['proleptic_gregorian', 'gregorian', 'standard']:
        logger.info(f"Converting {calendar} calendar to proleptic_gregorian calendar")
        ds_ar['time'] = convert_cftime_to_standard(ds_ar['time'].values)
    else:
        logger.info(f"Dataset already uses standard calendar: {calendar}")

    # Find common time range
    common_times = sorted(set(ds_mcs['time'].values).intersection(set(ds_ar['time'].values)))
    if not common_times:
        logger.warning("No common time values between datasets!")
        return None
    else:
        # Select only the common times in both datasets
        ds_mcs = ds_mcs.sel(time=common_times)
        ds_ar = ds_ar.sel(time=common_times)

    # Merge the datasets
    ds = xr.merge([ds_mcs, ds_ar], combine_attrs='drop_conflicts')
    logger.info(f"Successfully merged datasets with {len(common_times)} common time points")

    # Rename variables, drop unwanted ones in the DataSet
    ds = ds.rename(rename_dict).drop_vars(drop_var_list)

    # TODO: Modify global attributes if needed
    # ds.attrs['history'] = f"Created on {time.ctime()} by combining MCS and AR tracking data"
    
    # Write to Zarr
    if out_zarr:
        write_zarr(ds, ds_mcs, out_zarr, client=client, logger=logger)
    
    return ds

#-------------------------------------------------------------------
def write_zarr(ds, ds_mcs, out_zarr, client=None, logger=None):
    """
    Write dataset to Zarr with optimized chunking for HEALPix grid.
    
    Args:
        ds: xarray.Dataset
            Dataset to write
        ds_mcs: xarray.Dataset
            MCS dataset with HEALPix attributes
        out_zarr: str
            Output Zarr store path
        client: dask.distributed.Client, optional
            Dask client for distributed computation
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Optimize cell chunking for HEALPix grid
    zoom_level = zoom_level_from_nside(ds_mcs.crs.attrs['healpix_nside'])
    chunksize_time = 24
    chunksize_cell = 12 * 4**zoom_level
    
    # Make time chunks more even if needed
    if isinstance(chunksize_time, (int, float)) and chunksize_time != 'auto':
        total_times = ds.sizes['time']
        chunks = total_times // chunksize_time
        if chunks * chunksize_time < total_times:
            # We have a remainder - try to make chunks more even
            if total_times % chunks == 0:
                chunksize_time = total_times // chunks
            elif total_times % (chunks + 1) == 0:
                chunksize_time = total_times // (chunks + 1)
    
    # Set proper chunking for HEALPix output
    chunked_hp = ds.chunk({
        "time": chunksize_time, 
        "cell": chunksize_cell, 
    })
    # Report dataset size and chunking info
    logger.info(f"Output dataset dimensions: {dict(chunked_hp.sizes)}")
    logger.info(f"Output chunking scheme: time={chunksize_time}, cell={chunksize_cell}")

    # ---------- WRITE HEALPIX ZARR OUTPUT ----------
    logger.info(f"Starting Zarr write to: {out_zarr}")
    
    # Create a delayed task for Zarr writing
    write_task = chunked_hp.to_zarr(
        out_zarr,
        mode="w",        
        consolidated=True,  # Enable for better performance when reading
        compute=False      # Create a delayed task
    )
    
    # Compute the task, with progress reporting
    if client:
        from dask.distributed import progress
        import psutil

        # Temporarily suppress distributed.shuffle logs during progress display
        shuffle_logger = logging.getLogger('distributed.shuffle')
        original_level = shuffle_logger.level
        shuffle_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

        # Get cluster state information before processing
        memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
        logger.info(f"Current memory usage across workers (GB): {memory_usage}")
               
        try:
            # Compute with progress tracking
            future = client.compute(write_task)
            logger.info("Writing Zarr (this may take a while)...")
            progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals

            result = future.result()
            logger.info("Zarr write completed successfully")
        except Exception as e:
            logger.error(f"Zarr write failed: {str(e)}")
            raise
        finally:
            # Restore original log level
            shuffle_logger.setLevel(original_level)
    else:
        # Compute locally if no client
        write_task.compute()

    logger.info(f"Zarr file complete: {out_zarr}")

#-------------------------------------------------------------------
def convert_cftime_to_standard(cftime_times):
    """
    Convert cftime objects to pandas Timestamps (proleptic_gregorian calendar)
    
    Args:
        cftime_times: cftime object or array-like
            Single cftime datetime object or array of cftime datetime objects
    
    Returns:
        pandas.DatetimeIndex or pandas.Timestamp: 
            DatetimeIndex with proleptic_gregorian calendar if input is array-like,
            or a single Timestamp if input is a single cftime object
    """
    # Check if input is a single cftime object (has year attribute directly)
    is_single_object = hasattr(cftime_times, 'year')
    
    # If single object, convert it to a list with one element
    if is_single_object:
        cftime_list = [cftime_times]
    else:
        cftime_list = cftime_times
    
    # Extract date components from cftime objects
    timestamps = []
    for t in cftime_list:
        # Extract time components from the cftime object
        dt_components = {
            'year': t.year,
            'month': t.month,
            'day': t.day,
            'hour': t.hour if hasattr(t, 'hour') else 0,
            'minute': t.minute if hasattr(t, 'minute') else 0,
            'second': t.second if hasattr(t, 'second') else 0
        }
        # Create a pandas timestamp with the same components (proleptic_gregorian)
        timestamps.append(pd.Timestamp(**dt_components))
    
    # Return either a single Timestamp or a DatetimeIndex based on input type
    if is_single_object:
        return timestamps[0]
    else:
        return pd.DatetimeIndex(timestamps)

#-------------------------------------------------------------------
def zoom_level_from_nside(nside):
    """
    Calculate the zoom level from the NSIDE value.

    Args:
        nside (int): NSIDE value, must be a power of 2.
    
    Returns:
        int: Zoom level corresponding to the NSIDE value.
    """
    zoom = int(np.log2(nside))
    if 2**zoom != nside:
        raise ValueError("NSIDE must be a power of 2.")
    return zoom

def setup_logging():
    """
    Set the logging message level

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Starting NetCDF to HEALPix Zarr conversion: ...")

    zoom = 8
    version = 'v1'

    # Input directories and filenames
    dir_mcs = f"/pscratch/sd/w/wcmca1/scream-cess-healpix/mcs_tracking_hp9/mcstracking/scream2D_hrly_mcsmask_hp8_v1.zarr"
    dir_ar = f"/pscratch/sd/b/beharrop/kmscale_hackathon/hackathon_pre/"
    basename_ar = "scream2D_ne120_hp8_fast.ar_filtered_nodes.for_Lexie"

    # Output Zarr file
    out_dir = "/pscratch/sd/w/wcmca1/scream-cess-healpix/"
    out_basename = f"scream2D_allmasks_hp{zoom}_{version}.zarr"
    out_zarr = f"{out_dir}{out_basename}"

    parallel = False
    n_workers = 128
    threads_per_worker = 1

    if parallel:
        # Set Dask temporary directory for workers
        dask_tmp_dir = "/tmp"
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        client.run(setup_logging)

    # Get client if available
    try:
        from dask.distributed import get_client
        client = get_client()
        logger.info(f"Using existing Dask client with {len(client.scheduler_info()['workers'])} workers")
    except ValueError:
        logger.warning("No Dask client found, continuing without explicit client")
        client = None

    # ---------- FIND INPUT FILES ----------
    files_ar = sorted(glob.glob(f"{dir_ar}{basename_ar}*.nc"))
    logger.info(f"Number of AR files: {len(files_ar)}")

    # ---------- READ INPUT FILES ----------
    logger.info("Reading AR files...")
    # Open as a lazy dataset
    ds_ar = xr.open_mfdataset(
        files_ar,
        combine="by_coords",
        parallel=parallel,
        chunks={},
        mask_and_scale=False,
    )
    logger.info(f"Finished reading AR files.")

    logger.info("Reading MCS file...")
    # Open as a lazy dataset
    ds_mcs = xr.open_zarr(
        dir_mcs,
        consolidated=True,
        mask_and_scale=False,
    )
    logger.info(f"Finished reading MCS file")

    # Combine masks and write to Zarr
    ds = combine_masks(ds_mcs, ds_ar, client=client, out_zarr=out_zarr, logger=logger)

    # Log completion time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Combine completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")

    # import pdb; pdb.set_trace()