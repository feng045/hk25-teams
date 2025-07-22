import xarray as xr
import numpy as np
import os
import glob
import time
import logging
import intake
import requests
import easygems.healpix as egh
from functools import partial
from dask.distributed import Client, LocalCluster

def setup_logging():
    """
    Set the logging message level

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


def setup_dask_client(parallel, n_workers, threads_per_worker, memory_per_worker="60GB", logger=None):
    """
    Set up a Dask client optimized for HPC hardware
    
    Args:
        parallel: bool
            Whether to use parallel processing
        n_workers: int
            Number of workers for the Dask cluster
        threads_per_worker: int
            Number of threads per worker
        memory_per_worker: str
            Memory limit per worker (e.g., "60GB")
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
    
    logger.info(f"Setting up Dask cluster optimized for HPC hardware")
    logger.info(f"Workers: {n_workers}, Threads per worker: {threads_per_worker}")
    logger.info(f"Memory per worker: {memory_per_worker}")
    
    # # Enable NUMA-aware memory allocation
    # import os
    # os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    # os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
    # os.environ['NUMBA_NUM_THREADS'] = str(threads_per_worker)
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_per_worker,
        processes=True,  # Use processes for better memory isolation
        scheduler_port=0,
        dashboard_address=':8787',
        # Worker memory management settings
        memory_target_fraction=0.8,
        memory_spill_fraction=0.85,
        memory_pause_fraction=0.9,
        silence_logs=False,  # Keep logs for debugging
    )
    client = Client(cluster)
    
    # Configure client for high-throughput workloads
    try:
        client.configure({
            'distributed.worker.memory.target': 0.8,
            'distributed.worker.memory.spill': 0.85,
            'distributed.worker.memory.pause': 0.9,
            'distributed.worker.memory.terminate': 0.95,
            'distributed.comm.timeouts.tcp': '300s',
            'distributed.client.heartbeat': '10s',
            'distributed.worker.daemon': False,
        })
    except Exception as e:
        logger.warning(f"Could not configure some client settings: {e}")
    
    logger.info(f"Dask dashboard: {client.dashboard_link}")
    
    return client

def get_datasets(files_ar, files_tc, files_etc, parallel=False, logger=None):
    """
    Load datasets from files
    
    Args:
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
    
    def process_dataset(files, name):
        """Helper to process each dataset with consistent error handling"""
        logger.info(f"Reading {name} files...")
        
        # Process each file individually, then concatenate
        datasets = []
        for file in files:
            try:
                ds_single = xr.open_dataset(
                    file,
                    chunks={},
                    mask_and_scale=False,
                )
                datasets.append(ds_single)
            except Exception as e:
                logger.warning(f"Error opening {file}: {e}")
                continue
        
        if not datasets:
            raise ValueError(f"Could not open any {name} files")
        
        # Concatenate along time dimension
        ds = xr.concat(datasets, dim="time")
        
        # Sort time values to ensure monotonic order
        logger.info(f"Sorting {name} dataset by time")
        ds = ds.sortby('time')
        
        # Check for and remove duplicate time values
        _, index = np.unique(ds['time'].values, return_index=True)
        if len(index) < len(ds['time']):
            logger.warning(f"Found {len(ds['time']) - len(index)} duplicate time values in {name} files")
            ds = ds.isel(time=sorted(index))
    
        logger.info(f"Finished reading {name} files.")
        return ds
    
    # Process each dataset using our helper function
    ds_ar = process_dataset(files_ar, "AR")
    ds_tc = process_dataset(files_tc, "TC")
    ds_etc = process_dataset(files_etc, "ETC")

    return ds_ar, ds_tc, ds_etc

def combine_masks(ds_ar, ds_tc, ds_etc, logger=None):
    """
    Combine AR, TC, ETC tracking datasets.
    
    Args:
        ds_ar: xarray.Dataset
            AR tracking dataset
        ds_tc: xarray.Dataset
            TC tracking dataset
        ds_etc: xarray.Dataset
            ETC tracking dataset
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        xarray.Dataset: Combined dataset with all masks
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    drop_var_list = ['tp']
    rename_dict = {
        'AR_binary_tag': 'ar_mask',
        'TC_binary_tag': 'tc_mask',
        'ETC_binary_tag': 'etc_mask',
        'longitude': 'lon',
        'latitude': 'lat',
    }
    
    # Find common time range across all datasets
    common_times = sorted(set(ds_ar['time'].values)
                         .intersection(set(ds_tc['time'].values))
                         .intersection(set(ds_etc['time'].values)))
    if not common_times:
        logger.warning("No common time values between all datasets!")
        return None
    else:
        # Select only the common times in all datasets
        ds_ar = ds_ar.sel(time=common_times)
        ds_tc = ds_tc.sel(time=common_times)
        ds_etc = ds_etc.sel(time=common_times)

    # Fix for lat/lon coordinates issue: ensure consistent treatment
    datasets = [ds_ar, ds_tc, ds_etc]

    # Merge the datasets
    ds = xr.merge(datasets, combine_attrs='drop_conflicts', compat='override')
    logger.info(f"Successfully merged datasets with {len(common_times)} common time points")

    # Rename variables, drop unwanted ones in the DataSet
    ds = ds.rename(rename_dict).drop_vars(drop_var_list, errors='ignore')

    # TODO: Modify global attributes if needed
    # ds.attrs['history'] = f"Created on {time.ctime()} by combining tracking data"
    
    return ds


def fix_coords(ds, lat_dim="lat", lon_dim="lon", roll=False):
    """
    Fix coordinates in a dataset:
    1. Convert longitude from -180/+180 to 0-360 range (optional)
    2. Roll dataset to start at longitude 0 (optional)
    3. Ensure coordinates are in ascending order
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with lat/lon coordinates
    lat_dim : str, optional
        Name of latitude dimension, default "lat"
    lon_dim : str, optional
        Name of longitude dimension, default "lon"
    roll : bool, optional, default=False
        If True, convert longitude from -180/+180 to 0-360, and roll the dataset to start at longitude 0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset with fixed coordinates
    """
    if roll:
        # Find where longitude crosses from negative to positive (approx. where lon=0)
        lon_0_index = (ds[lon_dim] < 0).sum().item()
        
        # Create indexers for the roll
        lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)
        
        # Roll dataset and convert longitudes to 0-360 range
        ds = ds.isel({lon_dim: lon_indices})
        lon360 = xr.where(ds[lon_dim] < 0, ds[lon_dim] + 360, ds[lon_dim])
        ds = ds.assign_coords({lon_dim: lon360})
    
    # Ensure latitude and longitude are in ascending order if needed
    if np.all(np.diff(ds[lat_dim].values) < 0):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    if np.all(np.diff(ds[lon_dim].values) < 0):
        ds = ds.isel({lon_dim: slice(None, None, -1)})
    
    return ds


def is_valid(ds, tolerance=0.1):
    """
    Limit extrapolation distance to a certain tolerance.
    This is useful for preventing extrapolation of regional data to global HEALPix grid.

    Args:
        ds (xarray.Dataset):
            The dataset containing latitude and longitude coordinates.
        tolerance (float): default=0.1
            The maximum allowed distance in [degrees] for extrapolation.

    Returns:
        xarray.DataSet.
    """
    return (np.abs(ds.lat - ds.lat_hp) < tolerance) & (np.abs(ds.lon - ds.lon_hp) < tolerance)


def calculate_healpix_tolerance(zoom_level):
    """
    Calculate appropriate tolerance for is_valid function based on HEALPix zoom level.
    Returns approximately one grid cell size in degrees.
    
    Args:
        zoom_level (int): HEALPix zoom level
        
    Returns:
        float: Tolerance in degrees
    """
    # Calculate nside from zoom level (nside = 2^zoom)
    # nside determines HEALPix resolution - each increase in zoom doubles the resolution
    nside = 2 ** zoom_level
    
    # Calculate approximate pixel size in degrees
    # Mathematical derivation:
    # - Sphere has total area of 4π steradians (= 4π × (180/π)² sq. degrees)
    # - HEALPix divides sphere into 12 × nside² equal-area pixels
    # - Each pixel has area = 4π × (180/π)² / (12 × nside²) sq. degrees
    # - Linear size = √(pixel area) ≈ 58.6 / nside degrees
    # This gives approximately the angular width of one HEALPix cell
    pixel_size_degrees = 58.6 / nside
    
    return pixel_size_degrees

def remap_to_healpix_and_save(ds, catalog_dict, out_zarr, 
                                chunksize_cell, chunksize_time,
                                client=None, logger=None):
    """
    Remap a dataset to HEALPix grid and save as Zarr.
    
    Args:
        ds : xarray.Dataset
            Input dataset to remap.
        catalog_dict : dict
            Dictionary with catalog information.
        zoom : int
            HEALPix zoom level.
        out_zarr : str
            Output Zarr file path.
        chunksize_cell : int
            Chunk size for cell dimension.
        chunksize_time : int
            Chunk size for time dimension.
        client : (dask.distributed.Client, optional)
            Dask client.
        logger : (logging.Logger, optional)
            Logger for debug information.

    Returns:
        xarray.Dataset: The remapped HEALPix dataset
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Get catalog parameters
    catalog_file = catalog_dict["catalog_file"]
    catalog_location = catalog_dict["catalog_location"]
    catalog_source = catalog_dict["catalog_source"]
    catalog_params = catalog_dict["catalog_params"]
    zoom = catalog_params.get("zoom", 8)

    # Save longitude sign info
    signed_lon = True if np.min(ds["lon"]) < 0 else False
    logger.info(f"Input data lon coordinate has negative values: {signed_lon}")

    # Load the HEALPix catalog
    logger.info(f"Loading HEALPix catalog: {catalog_file}")
    in_catalog = intake.open_catalog(catalog_file)
    if catalog_location:
        in_catalog = in_catalog[catalog_location]
    
    # Get the DataSet from the catalog
    ds_hp = in_catalog[catalog_source](**catalog_params).to_dask()
    # Add lat/lon coordinates to the HEALPix DataSet
    ds_hp = ds_hp.pipe(partial(egh.attach_coords, signed_lon=signed_lon))
    
    # Assign extra coordinates (lon_hp, lat_hp) to the HEALPix coordinates
    # This is needed for limiting the extrapolation during remapping
    lon_hp = ds_hp.lon.assign_coords(cell=ds_hp.cell, lon_hp=lambda da: da)
    lat_hp = ds_hp.lat.assign_coords(cell=ds_hp.cell, lat_hp=lambda da: da)
    
    # Make sure coordinates are fixed before remapping
    ds = fix_coords(ds)

    # Calculate appropriate tolerance based on zoom level
    tolerance = calculate_healpix_tolerance(zoom)
    logger.info(f"Using HEALPix tolerance of {tolerance:.4f}° at zoom level {zoom}")
    
    # Remap DataSet to HEALPix
    logger.info("Applying nearest neighbor remapping to HEALPix grid...")
    fill_value = 0
    dsout_hp = ds.sel(
        lon=lon_hp, lat=lat_hp, method="nearest",
    ).where(partial(is_valid, tolerance=tolerance), fill_value)

    # Drop lat/lon coordinates (not needed in HEALPix)
    dsout_hp = dsout_hp.drop_vars(["lat_hp", "lon_hp", "lat", "lon"])
    # Update globle attributes
    dsout_hp.attrs['Title'] = f"HEALPix remapped tracking mask data (zoom={zoom})"
    dsout_hp.attrs['zoom'] = zoom
    dsout_hp.attrs["Created_on"] = time.ctime(time.time())
    
    # Set proper chunking for HEALPix output
    chunked_hp = dsout_hp.chunk({
        "time": chunksize_time, 
        "cell": chunksize_cell, 
    })
    
    # Report dataset size and chunking info
    logger.info(f"HEALPix dataset dimensions: {dict(chunked_hp.sizes)}")
    logger.info(f"HEALPix chunking scheme: time={chunksize_time}, cell={chunksize_cell}")
    
    # ---------- WRITE HEALPIX ZARR OUTPUT ----------
    logger.info(f"Starting HEALPix Zarr write to: {out_zarr}")
    
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
            logger.info("Writing HEALPix Zarr (this may take a while)...")
            progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals

            result = future.result()
            logger.info("HEALPix Zarr write completed successfully")
        except Exception as e:
            logger.error(f"HEALPix Zarr write failed: {str(e)}")
            raise
        finally:
            # Restore original log level
            shuffle_logger.setLevel(original_level)
    else:
        # Compute locally if no client
        write_task.compute()

    logger.info(f"HEALPix conversion complete: {out_zarr}")
    
    return dsout_hp

def main():
    """Main function to run the remap masks"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("Starting remap masks ...")

    # Define parameters
    source_name = "ERA5"
    zoom = 8
    version = "v1"
    parallel = True
    n_workers = 8
    threads_per_worker = 16
    memory_per_worker = "60GB"
    chunksize_cell = 12 * 4**zoom
    chunksize_time = 24

    catalog_dict = {
        "catalog_file": "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml",
        "catalog_location": "NERSC",
        "catalog_source": "IR_IMERG",
        "catalog_params": {"zoom": zoom},
    }

    in_dir = "/pscratch/sd/b/beharrop/kmscale_hackathon/ERA5_tracking/"
    dir_ar = f"{in_dir}"
    dir_tc = f"{in_dir}TC_files_PRECT/"
    dir_etc = f"{in_dir}ERA5_ETCtag/"
    basename_ar = f"ERA5_AR_tracks.nc"
    basename_tc = f"e5.accumulated_tp_6h.*.tc_filtered.nc"
    basename_etc = f"ERA5_ETCtag_*.nc"

    out_dir = "/pscratch/sd/w/wcmca1/hackathon/allmasks/"
    out_basename = f"{source_name}_AR_TC_ETC_hp{zoom}_{version}.zarr"
    out_zarr = f"{out_dir}{out_basename}"
    os.makedirs(out_dir, exist_ok=True)

    # Setup Dask client
    client = setup_dask_client(parallel, n_workers, threads_per_worker, memory_per_worker, logger)

    try:
        # Find input files
        files_ar = sorted(glob.glob(f"{dir_ar}{basename_ar}"))
        files_tc = sorted(glob.glob(f"{dir_tc}{basename_tc}"))
        files_etc = sorted(glob.glob(f"{dir_etc}{basename_etc}"))
        logger.info(f"Number of AR files: {len(files_ar)}")
        logger.info(f"Number of TC files: {len(files_tc)}")
        logger.info(f"Number of ETC files: {len(files_etc)}")

        # Load datasets
        ds_ar, ds_tc, ds_etc = get_datasets(files_ar, files_tc, files_etc, parallel, logger)

        # Combine datasets
        ds = combine_masks(ds_ar, ds_tc, ds_etc, logger=logger)
        # import pdb; pdb.set_trace()

        # Remap to HEALPix and save
        dsout_hp = remap_to_healpix_and_save(ds, catalog_dict, out_zarr, 
                                        chunksize_cell, chunksize_time,
                                        client=client, logger=logger)
        
        # Cleanup
        ds_ar.close()
        ds_tc.close()
        ds_etc.close()
        ds.close()
        dsout_hp.close()
        
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
    logger.info(f"Conversion completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")

if __name__ == "__main__":
    main()