import xarray as xr
import numpy as np
import pandas as pd
import os
import cftime
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
def combine_masks(ds_mcs, ds_ar, ds_tc, ds_etc, client=None, out_zarr=None, logger=None):
    """
    Combine MCS and AR tracking datasets and write to Zarr store.
    
    Args:
        ds_mcs: xarray.Dataset
            MCS tracking dataset
        ds_ar: xarray.Dataset
            AR tracking dataset
        ds_tc: xarray.Dataset
            TC tracking dataset
        ds_etc: xarray.Dataset
            ETC tracking dataset
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

    drop_var_list = ['ccs_mask', 'pr']
    rename_dict = {
        'AR_binary_tag': 'ar_mask',
        'TC_binary_tag': 'tc_mask',
        'ETC_binary_tag': 'etc_mask',
    }

    # Check the calendar type of the time coordinate in ds_ar
    calendar = ds_ar['time'].dt.calendar

    # Convert ds_mcs time coordinate to match ds_ar calendar
    if calendar not in ['proleptic_gregorian', 'gregorian', 'standard']:
        logger.info(f"Converting ds_mcs time from standard calendar to {calendar} calendar")
        converted_times = convert_to_matching_calendar(ds_mcs['time'].values, calendar)
        
        # Replace the time values in ds_mcs with the converted ones
        ds_mcs = ds_mcs.assign_coords(time=converted_times)
        logger.info(f"Successfully converted ds_mcs time to {calendar} calendar")
    else:
        logger.info(f"Both datasets use standard calendar: {calendar}, no conversion needed")
    
    # Find common time range across all three datasets
    common_times = sorted(set(ds_mcs['time'].values)
                         .intersection(set(ds_ar['time'].values))
                         .intersection(set(ds_tc['time'].values))
                         .intersection(set(ds_etc['time'].values)))
    if not common_times:
        logger.warning("No common time values between all datasets!")
        return None
    else:
        # Select only the common times in all datasets
        ds_mcs = ds_mcs.sel(time=common_times)
        ds_ar = ds_ar.sel(time=common_times)
        ds_tc = ds_tc.sel(time=common_times)
        ds_etc = ds_etc.sel(time=common_times)

    # Fix for lat/lon coordinates issue: ensure consistent treatment
    datasets = [ds_mcs, ds_ar, ds_tc, ds_etc]
    # Fix dimensions and coordinates for consistent merging
    for i, ds in enumerate(datasets):
        # Rename 'ncol' to 'cell' if it exists to standardize dimensions
        if 'ncol' in ds.dims:
            logger.info(f"Renaming dimension 'ncol' to 'cell' in dataset {i}")
            datasets[i] = ds.rename({'ncol': 'cell'})
        
        # Drop lat and lon variables from all datasets
        for var in ['lat', 'lon']:
            if var in ds.variables:
                logger.info(f"Dropping {var} from dataset {i}")
                datasets[i] = datasets[i].drop_vars(var)
    
    # Merge the datasets
    ds = xr.merge(datasets, combine_attrs='drop_conflicts', compat='override')
    logger.info(f"Successfully merged datasets with {len(common_times)} common time points")

    # Rename variables, drop unwanted ones in the DataSet
    ds = ds.rename(rename_dict).drop_vars(drop_var_list, errors='ignore')

    # TODO: Modify global attributes if needed
    # ds.attrs['history'] = f"Created on {time.ctime()} by combining MCS and AR tracking data"
    
    # Write to Zarr
    if out_zarr:
        write_zarr(ds, out_zarr, client=client, logger=logger)
    
    return ds

#-------------------------------------------------------------------
def write_zarr(ds, out_zarr, client=None, logger=None):
    """
    Write dataset to Zarr with optimized chunking for HEALPix grid.
    
    Args:
        ds: xarray.Dataset
            Dataset to write
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
    zoom_level = zoom_level_from_nside(ds.crs.attrs['healpix_nside'])
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
def convert_to_matching_calendar(std_times, target_calendar):
    """
    Convert standard calendar (proleptic_gregorian) timestamps to match a target calendar.
    
    Args:
        std_times: array of numpy.datetime64, pandas.DatetimeIndex or pandas.Timestamp
            Timestamps with standard (proleptic_gregorian) calendar
        target_calendar: str
            Target calendar to convert to ('365_day', '360_day', 'noleap', etc.)
            
    Returns:
        cftime.datetime objects using the target calendar
    """

    
    # Initialize the appropriate cftime date type based on target calendar
    calendar_types = {
        '365_day': cftime.DatetimeNoLeap,
        'noleap': cftime.DatetimeNoLeap,
        '360_day': cftime.Datetime360Day,
        'all_leap': cftime.DatetimeAllLeap,
        'julian': cftime.DatetimeJulian,
        # Add other calendars as needed
    }
    
    if target_calendar in ['proleptic_gregorian', 'gregorian', 'standard']:
        # No conversion needed
        return std_times
    
    if target_calendar not in calendar_types:
        raise ValueError(f"Unsupported calendar: {target_calendar}")
        
    datetime_type = calendar_types[target_calendar]
    
    # Check if input is a single timestamp
    is_single_object = not hasattr(std_times, '__iter__') or isinstance(std_times, pd.Timestamp)
    
    # Convert to list for uniform processing
    times_list = [std_times] if is_single_object else std_times
    
    # Convert each timestamp to the target calendar
    converted_times = []
    for t in times_list:
        # Convert numpy.datetime64 to pandas.Timestamp which has the necessary attributes
        if isinstance(t, np.datetime64):
            ts = pd.Timestamp(t)
            converted_times.append(datetime_type(
                ts.year, ts.month, ts.day, 
                ts.hour, ts.minute, ts.second
            ))
        else:
            # For pandas.Timestamp or datetime objects that already have year, month attributes
            converted_times.append(datetime_type(
                t.year, t.month, t.day, 
                t.hour, t.minute, t.second
            ))
    
    # Return a single object or a list based on input type
    if is_single_object:
        return converted_times[0]
    else:
        return converted_times

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


def setup_dask_client(parallel, n_workers, threads_per_worker, logger=None):
    """
    Set up a Dask client for parallel processing
    
    Args:
        parallel: bool
            Whether to use parallel processing
        n_workers: int
            Number of workers for the Dask cluster
        threads_per_worker: int
            Number of threads per worker
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        dask.distributed.Client or None: Dask client if parallel is True, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not parallel:
        logger.info("Running in sequential mode (parallel=False)")
        return None
    
    logger.info(f"Setting up Dask cluster with {n_workers} workers, {threads_per_worker} threads per worker")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='auto',
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")
    
    return client

def get_datasets(dir_mcs, files_ar, files_tc, files_etc, parallel=False, logger=None):
    """
    Load datasets from files
    
    Args:
        dir_mcs: str
            Directory containing MCS zarr store
        files_ar: list
            List of AR NetCDF files
        files_tc: list
            List of TC NetCDF files
        files_etc: list
            List of ETC NetCDF files
        parallel: bool
            Whether to use parallel processing
        logger: logging.Logger
            Logger for status messages
            
    Returns:
        tuple: (ds_mcs, ds_ar) datasets
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Read AR files
    logger.info("Reading AR files...")
    ds_ar = xr.open_mfdataset(
        files_ar,
        combine="by_coords",
        parallel=parallel,
        chunks={},
        mask_and_scale=False,
    )
    logger.info(f"Finished reading AR files.")

    # Read TC files
    logger.info("Reading TC files...")
    ds_tc = xr.open_mfdataset(
        files_tc,
        combine="by_coords",
        parallel=parallel,
        chunks={},
        mask_and_scale=False,
    )
    logger.info(f"Finished reading TC files.")

    # Read ETC files
    logger.info("Reading ETC files...")
    ds_etc = xr.open_mfdataset(
        files_etc,
        combine="by_coords",
        parallel=parallel,
        chunks={},
        mask_and_scale=False,
    )
    logger.info(f"Finished reading ETC files.")
    
    # Read MCS file
    logger.info("Reading MCS file...")
    ds_mcs = xr.open_zarr(
        dir_mcs,
        consolidated=True,
        mask_and_scale=False,
    )
    logger.info(f"Finished reading MCS file")
    
    return ds_mcs, ds_ar, ds_tc, ds_etc

def main():
    """Main function to run the mask combination process"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("Starting tracking mask combination...")
    
    # Configuration parameters
    zoom = 8
    version = 'v1'
    parallel = True
    n_workers = 32
    threads_per_worker = 4
    
    # Input/output paths
    dir_mcs = f"/pscratch/sd/w/wcmca1/scream-cess-healpix/mcs_tracking_hp9/mcstracking/scream2D_hrly_mcsmask_hp8_v1.zarr"
    # dir_ar = f"/pscratch/sd/b/beharrop/kmscale_hackathon/hackathon_pre/"
    # basename_ar = "scream2D_ne120_hp8_fast.ar_filtered_nodes.for_Lexie"
    # basename_tc = "scream2D_ne120_hp8_fast.tc_filtered_nodes.for_Lexie"
    # basename_etc = "scream2D_ne120_hp8_fast.etc_filtered_nodes.for_Lexie"
    # dir_ar = f"/pscratch/sd/b/beharrop/kmscale_hackathon/hackathon_pre/for_zhe/"
    dir_ar = f"/pscratch/sd/b/beharrop/kmscale_hackathon/hackathon_pre/scream_1year_test/"
    basename_ar = "AR_filt_nodes_scream2D_ne120_inst_ivt_hp8."
    basename_tc = "TC_filt_nodes_scream2D_ne120_inst_ivt_hp8."
    basename_etc = "ETC_filt_nodes_scream2D_ne120_inst_ivt_hp8."
    
    # Output paths
    out_dir = "/pscratch/sd/w/wcmca1/scream-cess-healpix/"
    out_basename = f"scream2D_allmasks_hp{zoom}_{version}.zarr"
    out_zarr = f"{out_dir}{out_basename}"
    
    # Setup Dask client
    client = setup_dask_client(parallel, n_workers, threads_per_worker, logger)
    
    try:
        # Find input files
        files_ar = sorted(glob.glob(f"{dir_ar}{basename_ar}*.nc"))
        files_tc = sorted(glob.glob(f"{dir_ar}{basename_tc}*.nc"))
        files_etc = sorted(glob.glob(f"{dir_ar}{basename_etc}*.nc"))
        logger.info(f"Number of AR files: {len(files_ar)}")
        logger.info(f"Number of TC files: {len(files_tc)}")
        logger.info(f"Number of ETC files: {len(files_etc)}")
        
        # Load datasets
        ds_mcs, ds_ar, ds_tc, ds_etc = get_datasets(dir_mcs, files_ar, files_tc, files_etc, parallel, logger)

        # Process and write output
        ds = combine_masks(ds_mcs, ds_ar, ds_tc, ds_etc, client=client, out_zarr=out_zarr, logger=logger)
        
        # Cleanup
        ds_mcs.close()
        ds_ar.close()
        if ds is not None:
            ds.close()
            
    finally:
        # Always cleanup client
        if client and parallel:
            logger.info("Shutting down Dask client")
            client.close()
    
    # Log completion time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Combine completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")

if __name__ == "__main__":
    main()