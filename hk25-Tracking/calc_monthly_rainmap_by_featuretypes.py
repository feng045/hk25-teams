import numpy as np
import sys, os
import xarray as xr
import pandas as pd
import time
import psutil
import argparse
import cftime
import intake
import requests
import logging
import easygems.healpix as egh
from dask.distributed import Client, LocalCluster, progress

def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Calculate monthly MCS precipitation statistics."
    )
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-s", "--start", help="first time to process, format=YYYY-mm-ddTHH", required=True)
    parser.add_argument("-e", "--end", help="last time to process, format=YYYY-mm-ddTHH", required=True)
    parser.add_argument("--zoom", help="HEALPix zoom level", type=int, default=None)
    parser.add_argument("--nworkers", help="number of Dask workers", type=int, default=12)
    parser.add_argument("--threads", help="threads per worker", type=int, default=10)
    parser.add_argument("--memory", help="memory limit per worker, e.g. '40GB' (default: auto)", default=None)
    parser.add_argument("--chunk_days", help="number of days to process in each chunk", type=int, default=5)
    parser.add_argument("--pcp_thresh", help="precipitation threshold in mm/h", type=float, default=2.0)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'config_file': args.config,
        'start_datetime': args.start,
        'end_datetime': args.end,
        'zoom': args.zoom,
        'n_workers': args.nworkers,
        'threads_per_worker': args.threads,
        'memory_limit': args.memory,
        'chunk_days': args.chunk_days,
        'pcp_thresh': args.pcp_thresh,
    }

    return args_dict

def get_memory_usage():
    """Get current memory usage in a human-readable format"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # Convert to GB
    memory_gb = memory_info.rss / (1024 ** 3)
    return memory_gb

def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

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

def process_month_chunked(month_ds, chunk_days=5, pcp_thresh=2.0):
    """Process one month of data in time chunks to reduce memory pressure"""
    # Current month's time value for output
    out_time = month_ds.time[0]
    out_time_str = out_time.dt.strftime('%Y-%m').item()
    _year = out_time.dt.year.item()
    _month = out_time.dt.month.item()
    _day = out_time.dt.day.item()
    # Create standard datetime using pandas
    std_time = pd.Timestamp(_year, _month, _day, 0, 0, 0)
    
    # Get total times in month
    ntimes = len(month_ds.time)
    
    # Detect time interval in hours
    if ntimes > 1:
        # Get the first two timestamps
        time_values = month_ds.time.values
        
        # Calculate time interval in hours based on type
        if isinstance(time_values[0], (cftime._cftime.DatetimeNoLeap, cftime._cftime.Datetime360Day)):
            # For cftime objects
            time_interval = (time_values[1] - time_values[0]).total_seconds() / 3600.0
        else:
            # For numpy datetime64 objects
            time_interval = (pd.Timestamp(time_values[1]) - pd.Timestamp(time_values[0])).total_seconds() / 3600.0
            
        print(f"  Detected time interval: {time_interval:.1f} hours")
    else:
        # Default to 1 hours if only one timestamp
        time_interval = 1.0
        print(f"  Using default time interval: {time_interval:.1f} hours")
    
    # Calculate steps per day based on time interval
    steps_per_day = 24.0 / time_interval
    
    # Initialize accumulators for summations
    # Total precipitation
    totprecip_sum = None
    
    # MCS statistics
    mcsprecip_sum = None
    mcscount_sum = None
    mcspcpct_sum = None
    
    # AR statistics
    arprecip_sum = None
    arcount_sum = None
    arpcpct_sum = None
    
    # TC statistics
    tcprecip_sum = None
    tccount_sum = None
    tcpcpct_sum = None
    
    # ETC statistics
    etcprecip_sum = None
    etccount_sum = None
    etcpcpct_sum = None
    
    # Process in chunks of days to limit memory usage
    step = int(chunk_days * steps_per_day)
    for start_idx in range(0, ntimes, step):
        end_idx = min(start_idx + step, ntimes)
        start_day = int(start_idx / steps_per_day) + 1
        end_day = int(end_idx / steps_per_day)
        print(f"  Processing days {start_day}-{end_day} of month {out_time_str}")
        
        # Extract chunk of data and compute immediately to free memory
        chunk_ds = month_ds.isel(time=slice(start_idx, end_idx)).compute()
        
        # Extract variables
        mcs_mask = chunk_ds['mcs_mask']
        ar_mask = chunk_ds['ar_mask']
        tc_mask = chunk_ds['tc_mask']
        etc_mask = chunk_ds['etc_mask']
        precipitation = chunk_ds['pr']
        
        # Compute total precipitation - multiply by time interval to get mm
        chunk_totprecip = (precipitation * time_interval).sum(dim='time')
        
        # Compute statistics for MCS - multiply by time interval to get mm
        chunk_mcsprecip = (precipitation.where(mcs_mask > 0) * time_interval).sum(dim='time')
        chunk_mcscount = (mcs_mask > 0).sum(dim='time')
        chunk_mcspcpct = (precipitation.where(mcs_mask > 0) > pcp_thresh).sum(dim='time')
        
        # Compute statistics for AR - multiply by time interval to get mm
        chunk_arprecip = (precipitation.where(ar_mask > 0) * time_interval).sum(dim='time')
        chunk_arcount = (ar_mask > 0).sum(dim='time')
        chunk_arpcpct = (precipitation.where(ar_mask > 0) > pcp_thresh).sum(dim='time')
        
        # Compute statistics for TC - multiply by time interval to get mm
        chunk_tcprecip = (precipitation.where(tc_mask > 0) * time_interval).sum(dim='time')
        chunk_tccount = (tc_mask > 0).sum(dim='time')
        chunk_tcpcpct = (precipitation.where(tc_mask > 0) > pcp_thresh).sum(dim='time')
        
        # Compute statistics for ETC - multiply by time interval to get mm
        chunk_etcprecip = (precipitation.where(etc_mask > 0) * time_interval).sum(dim='time')
        chunk_etccount = (etc_mask > 0).sum(dim='time')
        chunk_etcpcpct = (precipitation.where(etc_mask > 0) > pcp_thresh).sum(dim='time')
        
        # Accumulate results
        if totprecip_sum is None:
            # Initialize with first chunk results
            totprecip_sum = chunk_totprecip
            
            # MCS results
            mcsprecip_sum = chunk_mcsprecip
            mcscount_sum = chunk_mcscount
            mcspcpct_sum = chunk_mcspcpct
            
            # AR results
            arprecip_sum = chunk_arprecip
            arcount_sum = chunk_arcount
            arpcpct_sum = chunk_arpcpct
            
            # TC results
            tcprecip_sum = chunk_tcprecip
            tccount_sum = chunk_tccount
            tcpcpct_sum = chunk_tcpcpct
            
            # ETC results
            etcprecip_sum = chunk_etcprecip
            etccount_sum = chunk_etccount
            etcpcpct_sum = chunk_etcpcpct
        else:
            # Add subsequent chunk results
            totprecip_sum += chunk_totprecip
            
            # MCS results
            mcsprecip_sum += chunk_mcsprecip
            mcscount_sum += chunk_mcscount
            mcspcpct_sum += chunk_mcspcpct
            
            # AR results
            arprecip_sum += chunk_arprecip
            arcount_sum += chunk_arcount
            arpcpct_sum += chunk_arpcpct
            
            # TC results
            tcprecip_sum += chunk_tcprecip
            tccount_sum += chunk_tccount
            tcpcpct_sum += chunk_tcpcpct
            
            # ETC results
            etcprecip_sum += chunk_etcprecip
            etccount_sum += chunk_etccount
            etcpcpct_sum += chunk_etcpcpct
        
        # Explicitly delete chunk data to free memory
        del chunk_ds, mcs_mask, ar_mask, tc_mask, etc_mask, precipitation
        del chunk_mcsprecip, chunk_mcscount, chunk_mcspcpct
        del chunk_arprecip, chunk_arcount, chunk_arpcpct
        del chunk_tcprecip, chunk_tccount, chunk_tcpcpct
        del chunk_etcprecip, chunk_etccount, chunk_etcpcpct
        del chunk_totprecip
    
    # Return all statistics in a dictionary
    return {
        'time': std_time,
        'ntimes': ntimes,
        'time_interval': time_interval,
        'totprecip': totprecip_sum,
        
        # MCS statistics
        'mcsprecip': mcsprecip_sum,
        'mcscount': mcscount_sum,
        'mcspcpct': mcspcpct_sum,
        
        # AR statistics
        'arprecip': arprecip_sum,
        'arcount': arcount_sum,
        'arpcpct': arpcpct_sum,
        
        # TC statistics
        'tcprecip': tcprecip_sum,
        'tccount': tccount_sum,
        'tcpcpct': tcpcpct_sum,
        
        # ETC statistics
        'etcprecip': etcprecip_sum,
        'etccount': etccount_sum,
        'etcpcpct': etcpcpct_sum
    }


def write_netcdf(results, ds, output_filename, zoom, pcp_thresh, logger=None):
    """
    Write monthly precipitation statistics to a NetCDF file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f'Preparing data for output file: {output_filename}')
    
    # Prepare data for output dataset
    times = [r['time'] for r in results]
    ntimes_values = [r['ntimes'] for r in results]
    time_interval = results[0]['time_interval']
    
    # Extract total precipitation values
    totprecip_values = [r['totprecip'] for r in results]
    
    # Extract feature values for MCS, AR, TC, ETC
    mcsprecip_values = [r['mcsprecip'] for r in results]
    mcscount_values = [r['mcscount'] for r in results]
    mcspcpct_values = [r['mcspcpct'] for r in results]
    
    arprecip_values = [r['arprecip'] for r in results]
    arcount_values = [r['arcount'] for r in results]
    arpcpct_values = [r['arpcpct'] for r in results]
    
    tcprecip_values = [r['tcprecip'] for r in results]
    tccount_values = [r['tccount'] for r in results]
    tcpcpct_values = [r['tcpcpct'] for r in results]
    
    etcprecip_values = [r['etcprecip'] for r in results]
    etccount_values = [r['etccount'] for r in results]
    etcpcpct_values = [r['etcpcpct'] for r in results]

    # # Convert input datetimes to output datetime strings
    # try:
    #     # Handle both numpy datetime64 and cftime datetime objects
    #     if hasattr(ds.time.values[0], 'strftime'):
    #         # cftime objects
    #         start_datetime = ds.time.values[0].strftime('%Y%m%dT%H')
    #         end_datetime = ds.time.values[-1].strftime('%Y%m%dT%H')
    #     else:
    #         # numpy datetime64
    #         start_datetime = pd.Timestamp(ds.time.values[0]).strftime('%Y%m%dT%H')
    #         end_datetime = pd.Timestamp(ds.time.values[-1]).strftime('%Y%m%dT%H')
    # except Exception as e:
    #     logger.warning(f"Error formatting datetime: {e}")
    #     # Fallback
    #     start_datetime = str(ds.time.values[0])
    #     end_datetime = str(ds.time.values[-1])

    # Create output variables
    var_dict = {
        'precipitation': (['time', 'cell'], np.stack([r.values for r in totprecip_values])),
        'ntimes': (['time'], np.array(ntimes_values)),
        
        # MCS variables
        'mcs_precipitation': (['time', 'cell'], np.stack([r.values for r in mcsprecip_values])),
        'mcs_count': (['time', 'cell'], np.stack([r.values for r in mcscount_values])),
        'mcs_precipitation_count': (['time', 'cell'], np.stack([r.values for r in mcspcpct_values])),
        
        # AR variables
        'ar_precipitation': (['time', 'cell'], np.stack([r.values for r in arprecip_values])),
        'ar_count': (['time', 'cell'], np.stack([r.values for r in arcount_values])),
        'ar_precipitation_count': (['time', 'cell'], np.stack([r.values for r in arpcpct_values])),
        
        # TC variables
        'tc_precipitation': (['time', 'cell'], np.stack([r.values for r in tcprecip_values])),
        'tc_count': (['time', 'cell'], np.stack([r.values for r in tccount_values])),
        'tc_precipitation_count': (['time', 'cell'], np.stack([r.values for r in tcpcpct_values])),
        
        # ETC variables
        'etc_precipitation': (['time', 'cell'], np.stack([r.values for r in etcprecip_values])),
        'etc_count': (['time', 'cell'], np.stack([r.values for r in etccount_values])),
        'etc_precipitation_count': (['time', 'cell'], np.stack([r.values for r in etcpcpct_values])),
    }
    
    # Create coordinates
    coord_dict = {
        # Use time objects directly, not their .values
        'time': times,
        'cell': (['cell'], ds['cell'].values),
        'lat': (['cell'], ds['lat'].values),
        'lon': (['cell'], ds['lon'].values),
    }
    
    # Add crs if available
    if 'crs' in ds:
        coord_dict['crs'] = ds['crs'].values
    
    # Create global attributes
    gattr_dict = {
        'Title': 'Monthly precipitation statistics by feature type',
        'contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        # 'start_date': start_datetime,
        # 'end_date': end_datetime,
        'created_on': time.ctime(time.time()),
        'grid_type': 'HEALPix',
        'zoom_level': zoom,
        'time_interval': time_interval,
        'precipitation_threshold': pcp_thresh,
    }

    # Create output dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Add variable attributes
    dsout['cell'].attrs['long_name'] = 'HEALPix cell index'
    dsout['lon'].attrs['long_name'] = 'Longitude'
    dsout['lon'].attrs['units'] = 'degree'
    dsout['lat'].attrs['long_name'] = 'Latitude'
    dsout['lat'].attrs['units'] = 'degree'
    dsout['ntimes'].attrs['long_name'] = 'Number of hours during the month'
    dsout['ntimes'].attrs['units'] = 'count'
    
    # Total precipitation
    dsout['precipitation'].attrs['long_name'] = 'Total precipitation'
    dsout['precipitation'].attrs['units'] = 'mm'
    
    # MCS attributes
    dsout['mcs_precipitation'].attrs['long_name'] = 'MCS precipitation'
    dsout['mcs_precipitation'].attrs['units'] = 'mm'
    dsout['mcs_count'].attrs['long_name'] = 'Number of hours MCS is present'
    dsout['mcs_count'].attrs['units'] = 'hour'
    dsout['mcs_precipitation_count'].attrs['long_name'] = 'Number of hours MCS precipitation exceeds threshold'
    dsout['mcs_precipitation_count'].attrs['units'] = 'hour'
    
    # AR attributes
    dsout['ar_precipitation'].attrs['long_name'] = 'AR precipitation'
    dsout['ar_precipitation'].attrs['units'] = 'mm'
    dsout['ar_count'].attrs['long_name'] = 'Number of hours AR is present'
    dsout['ar_count'].attrs['units'] = 'hour'
    dsout['ar_precipitation_count'].attrs['long_name'] = 'Number of hours AR precipitation exceeds threshold'
    dsout['ar_precipitation_count'].attrs['units'] = 'hour'
    
    # TC attributes
    dsout['tc_precipitation'].attrs['long_name'] = 'TC precipitation'
    dsout['tc_precipitation'].attrs['units'] = 'mm'
    dsout['tc_count'].attrs['long_name'] = 'Number of hours TC is present'
    dsout['tc_count'].attrs['units'] = 'hour'
    dsout['tc_precipitation_count'].attrs['long_name'] = 'Number of hours TC precipitation exceeds threshold'
    dsout['tc_precipitation_count'].attrs['units'] = 'hour'
    
    # ETC attributes
    dsout['etc_precipitation'].attrs['long_name'] = 'ETC precipitation'
    dsout['etc_precipitation'].attrs['units'] = 'mm'
    dsout['etc_count'].attrs['long_name'] = 'Number of hours ETC is present'
    dsout['etc_count'].attrs['units'] = 'hour'
    dsout['etc_precipitation_count'].attrs['long_name'] = 'Number of hours ETC precipitation exceeds threshold'
    dsout['etc_precipitation_count'].attrs['units'] = 'hour'

    # Save the output file
    fillvalue = np.nan
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    logger.info(f'Writing output file: {output_filename}')
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='time', encoding=encoding)
    logger.info(f'Successfully wrote: {output_filename}')
    
    return dsout


def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Start timing
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")

    # # Get the command-line arguments
    # args_dict = parse_cmd_args()
    # config_file = args_dict.get('config_file')
    # start_datetime = args_dict.get('start_datetime')
    # end_datetime = args_dict.get('end_datetime')
    # zoom = args_dict.get('zoom')
    # n_workers = args_dict.get('n_workers')
    # threads_per_worker = args_dict.get('threads_per_worker')
    # memory_limit = args_dict.get('memory_limit')
    # chunk_days = args_dict.get('chunk_days')
    # pcp_thresh = args_dict.get('pcp_thresh')

    chunk_days = 6
    pcp_thresh = 0.1  # mm/h

    # Configuration parameters
    zoom = 8
    version = 'v1'
    parallel = True
    n_workers = 13
    threads_per_worker = 4

    # Catalog parameters
    catalog_file = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
    catalog_location = "NERSC"
    catalog_source = "scream_ne120"
    # Catalog parameters, can have multiple entries
    catalog_params = {"zoom": zoom}
    varname_precip_liq = "pr"   # Liquid precipitation variable name
    varname_precip_ice = "prs"  # Ice precipitation variable name (if exists)
    pr_convert_factor = 3600000.  # Convert precipitation flux from [m/s] to [mm/h]

    # Input combined mask file
    in_dir = "/pscratch/sd/w/wcmca1/scream-cess-healpix/"
    in_basename = f"scream2D_allmasks_hp{zoom}_{version}.zarr"
    in_zarr = f"{in_dir}{in_basename}"

    output_dir = "/pscratch/sd/w/wcmca1/scream-cess-healpix/monthly/"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}monthly_rainmap_by_featuretypes_hp{zoom}_{version}.nc"

    # Setup Dask client
    client = setup_dask_client(parallel, n_workers, threads_per_worker, logger)

    # Load the mask dataset
    ds = xr.open_zarr(in_zarr, consolidated=True)
    ds = ds.pipe(egh.attach_coords)

    # # Check catalog file availability
    # if catalog_file:
    #     if catalog_file.startswith(('http://', 'https://')):
    #         # Handle URL case
    #         try:
    #             response = requests.head(catalog_file, timeout=10)
    #             if response.status_code >= 400:
    #                 print(f"Catalog URL {catalog_file} returned status code {response.status_code}. Skipping remap.")
    #                 sys.exit('Code will exit now.')
    #         except requests.exceptions.RequestException as e:
    #             print(f"Error accessing catalog URL {catalog_file}: {str(e)}. Skipping remap.")
    #             sys.exit('Code will exit now.')
    #     elif os.path.isfile(catalog_file) is False:
    #         # Handle local file case
    #         print(f"Catalog file {catalog_file} does not exist. Skipping remap.")
    #         sys.exit('Code will exit now.')
    # else:
    #     print("Catalog file not specified in config. HEALPix remapping requires a catalog.")
    #     sys.exit('Code will exit now.')
    
    # Load the HEALPix catalog
    print(f"Loading HEALPix catalog: {catalog_file}")
    in_catalog = intake.open_catalog(catalog_file)
    if catalog_location:
        in_catalog = in_catalog[catalog_location]
    
    # Get the DataSet from the catalog
    ds_p = in_catalog[catalog_source](**catalog_params).to_dask()
    # Add lat/lon coordinates to the HEALPix DataSet
    ds_p = ds_p.pipe(egh.attach_coords)

    # Check liquid precipitaiton variable
    if varname_precip_liq in list(ds_p.keys()):
        # Convert liquid precipitation to mm/h
        pr = ds_p[varname_precip_liq] * pr_convert_factor
    # Check if the ice precipitation variable exist in the dataset
    if varname_precip_ice in list(ds_p.keys()):
        # Convert ice precipitation to liquid equivalent
        prs = ds_p[varname_precip_ice] * pr_convert_factor
        # Add ice precipitation to get total precipitation
        pr = pr + prs

    # Calendar conversion - check and convert calendars to match
    logger.info("Checking time coordinate calendars...")
    
    # Determine calendar types
    ds_p_calendar_type = type(ds_p.time.values[0]).__name__
    ds_calendar_type = type(ds.time.values[0]).__name__
    
    logger.info(f"Dataset calendars: ds_p uses {ds_p_calendar_type}, ds uses {ds_calendar_type}")
    
    # Convert ds time to match ds_p if they differ
    if ds_calendar_type != ds_p_calendar_type:
        logger.info(f"Converting ds time from {ds_calendar_type} to {ds_p_calendar_type}")
        
        # Get the calendar details from ds_p
        has_year_zero = True
        if hasattr(ds_p.time.values[0], 'has_year_zero'):
            has_year_zero = ds_p.time.values[0].has_year_zero
        
        # Convert datetime64 values to cftime DatetimeNoLeap objects
        new_times = []
        for t in ds.time.values:
            # Convert numpy datetime64 to pandas Timestamp to get date components
            pd_time = pd.Timestamp(t)
            # Create a matching cftime object
            dt_cftime = cftime.DatetimeNoLeap(
                pd_time.year, pd_time.month, pd_time.day,
                pd_time.hour, pd_time.minute, pd_time.second,
                has_year_zero=has_year_zero
            )
            new_times.append(dt_cftime)
        
        # Create a new dataset with the converted time coordinate
        ds = ds.assign_coords(time=new_times)
        logger.info("Calendar conversion complete")

    # Find common time range across all three datasets
    common_times = sorted(set(ds_p['time'].values)
                         .intersection(set(ds['time'].values)))
    if not common_times:
        logger.warning("No common time values between all datasets!")
        return None
    else:
        # Select only the common times in all datasets
        pr = pr.sel(time=common_times)
        ds = ds.sel(time=common_times)
        # Add precipitation to the dataset
        ds["pr"] = pr
    # import pdb; pdb.set_trace()

    # Group by month and apply the processing function
    monthly_results = []
    monthly_groups = ds.resample(time='1MS')

    # Check if client exists for parallel processing
    if client is not None:
        # Parallel processing with Dask
        logger.info("Running in parallel mode with Dask")
        delayed_results = []
        for month_start, month_ds in monthly_groups:
            print(f"Processing month: {month_start.strftime('%Y-%m')}")
            # Submit the processing job to the dask cluster
            delayed_result = client.submit(process_month_chunked, month_ds, chunk_days=chunk_days, pcp_thresh=pcp_thresh)
            delayed_results.append(delayed_result)
        
        # Clear line and show progress tracking
        print("\nTracking progress of all months processing in parallel:")
        progress(delayed_results)
        
        # Gather results (will wait for completion)
        results = client.gather(delayed_results)
    else:
        # Serial processing
        logger.info("Running in serial mode")
        results = []
        for month_start, month_ds in monthly_groups:
            print(f"Processing month: {month_start.strftime('%Y-%m')}")
            # Process directly without Dask
            result = process_month_chunked(month_ds, chunk_days=chunk_days, pcp_thresh=pcp_thresh)
            results.append(result)

    # Write output to NetCDF file
    write_netcdf(results, ds, output_filename, zoom, pcp_thresh, logger)


if __name__ == "__main__":
    main()