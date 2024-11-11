import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import os
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function Definitions

def read_data(max_distance, remove):
    """
    Reads the scattering coefficients dataset and processes it based on specified parameters.

    Parameters:
    -----------
    max_distance : float
        Maximum allowed distance for filtering samples.
    remove : bool
        Whether to remove scattering coefficients where frequency_0 < frequency_1.

    Returns:
    --------
    features : np.ndarray
        The processed feature matrix.
    species_labels : np.ndarray
        The species labels for each sample.
    station_labels : np.ndarray
        The station labels for each sample.
    distance_labels : np.ndarray
        The distance labels for each sample.
    """
    logging.info(f'Reading dataset with max_distance={max_distance} and remove={remove}')

    # READ THE STATIONS AND SPECIES/LABELS FROM THE DATASET
    with xr.open_dataset('../data/scattering_coefficients_32_32.nc') as dset:
        scat_coef_1 = dset['scat_coef_1'].mean(dim='comp')
        scat_coef_1 /= scat_coef_1.max(dim=('frequency_0','frequency_1'))
        species_labels = dset['species'].to_numpy()
        station_labels = dset['station'].to_numpy()
        distance_labels = dset['distance'].to_numpy()

    if remove:
        features = delete_scattering_coefficients(scat_coef_1)
    else:
        features = scat_coef_1.data.reshape(scat_coef_1.shape[0], -1)
    
    features, species_labels, station_labels, distance_labels = distance_thresholding(
        features, species_labels, station_labels, distance_labels, max_distance
    )

    logging.info('Data read and processed successfully')
    return features, species_labels, station_labels, distance_labels

def delete_scattering_coefficients(scat_coef_1):
    """
    Removes scattering coefficients where frequency_0 < frequency_1 and reshapes the data.

    Parameters:
    -----------
    scat_coef_1 : xarray.DataArray
        The scattering coefficients dataset.

    Returns:
    --------
    features : np.ndarray
        Processed and reshaped feature matrix.
    """
    logging.info('Removing scattering coefficients where frequency_0 < frequency_1')

    masking_matrix = np.ones(scat_coef_1.shape)
    for i, f_0 in enumerate(scat_coef_1.frequency_0):
        for j, f_1 in enumerate(scat_coef_1.frequency_1):
            if f_0 < f_1:
                masking_matrix[:, i, j] = np.nan

    scat_coef_1 = scat_coef_1 * masking_matrix
    features = scat_coef_1.data.reshape(scat_coef_1.shape[0], -1)
    mask = np.isnan(features).any(axis=0)
    features = features[:, ~mask]

    logging.info(f'Features reshaped to: {features.shape}')
    return features

def distance_thresholding(features, species_labels, station_labels, distance_labels, max_distance):
    """
    Filters the dataset based on a distance threshold.

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix.
    species_labels : np.ndarray
        Species labels.
    station_labels : np.ndarray
        Station labels.
    distance_labels : np.ndarray
        Distance labels.
    max_distance : float
        Maximum distance threshold.

    Returns:
    --------
    Tuple of filtered features, species_labels, station_labels, distance_labels.
    """
    logging.info(f'Applying distance threshold: max_distance={max_distance}')

    mask = distance_labels <= max_distance
    return features[mask], species_labels[mask], station_labels[mask], distance_labels[mask]

def stratified_train_test_split_with_station(features, y, station_labels, species2keep, max_distance, remove, test_size=0.2):
    """
    Splits the dataset into stratified train and test sets while keeping the station information.

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix.
    y : np.ndarray
        Species labels.
    station_labels : np.ndarray
        Station labels.
    species2keep : list
        Species to include in the split.
    max_distance : float
        Maximum allowed distance for filtering samples.
    remove : bool
        Whether to remove scattering coefficients where frequency_0 < frequency_1.
    test_size : float, optional
        Proportion of data to be used as the test set, default is 0.2.

    Returns:
    --------
    None
    """
    logging.info('Starting stratified train-test split')

    unique_station_labels = np.unique(station_labels)
    X_train, X_test, y_train, y_test = [], [], [], []
    station_labels_train, station_labels_test = [], []

    for species in species2keep:
        where_species = y == species
        for station in unique_station_labels:
            where_station = station_labels == station
            where = np.logical_and(where_species, where_station)

            if np.sum(where) == 0:
                continue

            features_where = features[where]
            y_where = y[where]
            data_len = len(features_where)

            train_size = int(data_len * (1 - test_size))

            X_train.append(features_where[:train_size])
            X_test.append(features_where[train_size:])
            y_train.append(y_where[:train_size])
            y_test.append(y_where[train_size:])
            station_labels_train.append(np.full(train_size, station))
            station_labels_test.append(np.full(data_len - train_size, station))

    # Concatenate lists into arrays
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    station_labels_train = np.concatenate(station_labels_train, axis=0)
    station_labels_test = np.concatenate(station_labels_test, axis=0)

    logging.info(f'Training set size: {len(y_train)} samples')
    logging.info(f'Test set size: {len(y_test)} samples')
    logging.info(f'Number of features: {X_train.shape[1]}')

    path2save = f'../data/svc/distance_cases/{str(max_distance).zfill(3)}/'
    os.makedirs(path2save, exist_ok=True)

    np.savez_compressed(f'{path2save}data_{max_distance}m.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, station_labels_train=station_labels_train, station_labels_test=station_labels_test)

# Constants
MAX_DISTANCE = 150
REMOVE = True
SPECIES2KEEP = ['elephant', 'giraffe', 'zebra', 'hyena']

# Main Execution
if __name__ == '__main__':
    logging.info('Script started')
    
    features, y, station_labels, distance_labels = read_data(MAX_DISTANCE, REMOVE)
    stratified_train_test_split_with_station(features, y, station_labels, SPECIES2KEEP, MAX_DISTANCE, REMOVE, test_size=0.2)

    logging.info('Script finished')
