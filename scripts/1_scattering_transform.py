# %%
import numpy as np
import pywt
import xarray as xr
from scipy.signal.windows import tukey
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# %%
def load_data(filepath):
    """
    Load the dataset from a NetCDF file.

    Parameters:
    filepath (str): Path to the NetCDF file.

    Returns:
    xr.Dataset: The loaded dataset.
    """
    
    return xr.open_dataset(filepath+'modified_dataset_glic.nc')

def normalize_chunks(dset):
    """
    Normalize the 'chunk' variable in the dataset by its maximum value along the 'time' dimension.

    Parameters:
    dset (xr.Dataset): The input dataset containing 'chunk' variable.

    Returns:
    xr.DataArray: The normalized 'chunk' variable.
    """
    norm_value = dset['chunk'].max(dim='time')
    dset['chunk'] /= norm_value

    return dset['chunk']

def define_wavelet_transform(fs, bins_0, bins_1):
    """
    Define the wavelet transform parameters including mother wavelet, frequencies, and scales.

    Parameters:
    fs (int): Sampling rate.
    bins_0 (int): Number of bins for the first wavelet transform.
    bins_1 (int): Number of bins for the second wavelet transform.

    Returns:
    dict: Dictionary containing wavelet parameters for both transforms.
    """
    mother_wavelet_0 = 'cmor1.0-1.0'
    frequencies_0 = np.geomspace(5, 90, num=bins_0) / fs
    scales_0 = pywt.frequency2scale(mother_wavelet_0, frequencies_0)

    mother_wavelet_1 = 'cmor2.0-1.0'
    frequencies_1 = np.geomspace(0.5, 80, num=bins_1) / fs
    scales_1 = pywt.frequency2scale(mother_wavelet_1, frequencies_1)

    return {
        'mother_wavelet_0': mother_wavelet_0,
        'scales_0': scales_0,
        'frequencies_0': frequencies_0 * fs,
        'mother_wavelet_1': mother_wavelet_1,
        'scales_1': scales_1,
        'frequencies_1': frequencies_1 * fs
    }

def process_batch(data_batch, wavelet_params, fs, alpha, start_idx):
    """
    Process a single batch of data to calculate scattering coefficients.

    Parameters:
    data_batch (xr.DataArray): A batch of the normalized data.
    wavelet_params (dict): The wavelet parameters.
    fs (int): Sampling rate.
    alpha (float): Parameter for the Tukey window.
    start_idx (int): Starting index of the batch in the original data.

    Returns:
    tuple: Scattering coefficients of the first and second order for the batch.
    """
    n_obs, n_channels, n_pts = data_batch.shape
    bins_0 = wavelet_params['scales_0'].shape[0]
    bins_1 = wavelet_params['scales_1'].shape[0]

    batch_scat_coef_0 = np.empty((n_obs, n_channels, bins_0))
    batch_scat_coef_1 = np.empty((n_obs, n_channels, bins_0, bins_1))

    # Apply Tukey window
    tukey_window = tukey(n_pts, alpha=alpha)
    batch_data_windowed = data_batch * tukey_window

    # First order scattering
    scalogram_0, _ = pywt.cwt(batch_data_windowed, wavelet_params['scales_0'], wavelet_params['mother_wavelet_0'], sampling_period=1/fs)
    scalogram_0 = np.abs(scalogram_0)
    batch_scat_coef_0 = np.mean(scalogram_0, axis=-1).transpose((1, 2, 0))

    # Second order scattering
    for i in range(bins_0):
        scalogram_1, _ = pywt.cwt(scalogram_0[i], wavelet_params['scales_1'], wavelet_params['mother_wavelet_1'], sampling_period=1/fs)
        scalogram_1 = np.abs(scalogram_1)
        batch_scat_coef_1[:, :, i, :] = np.mean(scalogram_1, axis=-1).transpose((1, 2, 0))

    return start_idx, batch_scat_coef_0, batch_scat_coef_1

def calculate_scattering_coefficients(data, wavelet_params, fs, alpha=0.1, batch_size=32, num_workers=6):
    """
    Calculate the scattering coefficients for the given data using the specified wavelet parameters.

    Parameters:
    data (xr.DataArray): The normalized data.
    wavelet_params (dict): The wavelet parameters.
    fs (int): Sampling rate.
    alpha (float): Parameter for the Tukey window.
    batch_size (int): Size of the batches for processing.
    num_workers (int): Number of worker processes to use.

    Returns:
    tuple: Tuple containing scattering coefficients of the first and second order.
    """
    n_obs, n_channels, n_pts = data.shape
    bins_0 = wavelet_params['scales_0'].shape[0]
    bins_1 = wavelet_params['scales_1'].shape[0]

    scat_coef_0 = np.empty((n_obs, n_channels, bins_0))
    scat_coef_1 = np.empty((n_obs, n_channels, bins_0, bins_1))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start_idx in range(0, n_obs, batch_size):
            end_idx = min(start_idx + batch_size, n_obs)
            batch_data = data[start_idx:end_idx]
            futures.append(executor.submit(process_batch, batch_data, wavelet_params, fs, alpha, start_idx))

        for future in tqdm(futures, desc="Processing batches", total=len(futures)):
            start_idx, batch_scat_coef_0, batch_scat_coef_1 = future.result()
            end_idx = start_idx + batch_scat_coef_0.shape[0]
            scat_coef_0[start_idx:end_idx] = batch_scat_coef_0
            scat_coef_1[start_idx:end_idx] = batch_scat_coef_1

    return scat_coef_0, scat_coef_1

def add_scattering_coefficients_to_dset(dset, wavelet_params, scat_coef_0, scat_coef_1):
    """Write scattering coefficients into the dataset.

    Args:
        dset: The dataset to write coefficients into.
        wavelet_params: The wavelet parameters.
        scat_coef_0: The first order scattering coefficients.
        scat_coef_1: The second order scattering coefficients.

    Returns:
        xarray.Dataset: Updated dataset with scattering coefficients.
    """

    # drop camera and waveform data
    dset = dset.drop_vars(('chunk'))

    dset['scat_coef_0'] = (['traces', 'comp', 'frequency_0'], scat_coef_0)
    dset['scat_coef_1'] = (['traces', 'comp', 'frequency_0', 'frequency_1'], scat_coef_1)

    # Assign frequency coordinates
    dset = dset.assign_coords(frequency_0=wavelet_params['frequencies_0'])
    dset = dset.assign_coords(frequency_1=wavelet_params['frequencies_1'])

    return dset

def save_dataset(dset, savename, path2data):
    """Save the updated dataset and the scattering network.

    Args:
        dset: The dataset to save.
    """
    dset.to_netcdf(f'{path2data}{savename}.nc')

# %%
# global variables
PATH2DATA = '/home/steinre/seissavanna/scattering_transform/'
NUM_WORKERS = 24
BATCH_SIZE = 64
FS = 200  # Sampling rate
BINS_0, BINS_1 = 32, 32
SAVENAME = f'scattering_coefficients_{BINS_0}_{BINS_1}'

# %%
# load and normalize data
dset = load_data(PATH2DATA)
#waveforms = normalize_chunks(dset)
waveforms = dset['chunk']

# Define wavelet transform parameters
wavelet_params = define_wavelet_transform(FS, BINS_0, BINS_1)

# %%
# Calculate scattering coefficients
scat_coef_0, scat_coef_1 = calculate_scattering_coefficients(waveforms, wavelet_params, FS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# %%
# Add scattering coefficients to dataset
dset = add_scattering_coefficients_to_dset(dset, wavelet_params, scat_coef_0, scat_coef_1)

# Save the updated dataset
save_dataset(dset, SAVENAME, PATH2DATA)


