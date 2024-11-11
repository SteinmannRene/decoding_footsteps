# %%
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
import ast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
def read_features_and_labels(file_path):
    """
    Read the feature map from a NetCDF file and process scattering coefficients.

    Parameters:
    - file_path: Path to the NetCDF file.

    Returns:
    - features: Numpy array of features.
    - labels: DataFrame containing species, station, and distance labels.
    """
    logging.info(f"Reading features and labels from {file_path}")
    with xr.open_dataset(file_path) as dset:
        scat_coef_1 = dset['scat_coef_1'].mean(dim='comp')
        scat_coef_1 /= scat_coef_1.max(dim=('frequency_0', 'frequency_1'))
        species_labels = dset['species'].to_numpy()
        station_labels = dset['station'].to_numpy()
        distance_labels = dset['distance'].to_numpy()

    features = scat_coef_1.data.reshape(scat_coef_1.shape[0], -1)
    labels = pd.DataFrame({
        'species': species_labels,
        'station': station_labels,
        'distance': distance_labels
    })
    logging.info("Features and labels successfully read and processed")
    return features, labels


def filter_species(labels, features, species_list):
    """
    Filter the features and labels to include only specified species.

    Parameters:
    - labels: DataFrame containing labels.
    - features: Numpy array of features.
    - species_list: List of species to include.

    Returns:
    - filtered_features: Numpy array of filtered features.
    - filtered_labels: DataFrame of filtered labels.
    """
    logging.info(f"Filtering data for species: {species_list}")
    species_list = sorted(species_list)
    indices = np.concatenate([np.where(labels['species'] == s)[0] for s in species_list])
    filtered_features = features[indices]
    filtered_labels = labels.iloc[indices]
    logging.info(f"Filtered down to {len(filtered_labels)} samples for species {species_list}")
    return filtered_features, filtered_labels


def filter_by_distance(features, labels, max_distance):
    """
    Filter the features and labels by a maximum distance.

    Parameters:
    - features: Numpy array of features.
    - labels: DataFrame containing labels.
    - max_distance: Maximum distance to filter by.

    Returns:
    - filtered_features: Numpy array of filtered features.
    - filtered_labels: DataFrame of filtered labels.
    """
    logging.info(f"Filtering data by max distance: {max_distance} meters")
    filtered_features = features[labels['distance'] < max_distance]
    filtered_labels = labels[labels['distance'] < max_distance]
    logging.info(f"Filtered down to {len(filtered_labels)} samples within {max_distance} meters")
    return filtered_features, filtered_labels


def split_train_test(features, labels, station, training_mode, unique_species):
    """
    Split the features and labels into training and testing sets based on station.

    Parameters:
    - features: Numpy array of features.
    - labels: DataFrame containing labels.
    - station: Station label to split by.
    - training_mode: 'normal' or 'transfer' mode to determine data splitting.
    - unique_species: List of unique species for splitting.

    Returns:
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    """
    logging.info(f"Splitting data for station {station} using mode {training_mode}")
    where_station = station == labels['station']
    y_station = labels['species'][where_station]
    X_station = features[where_station]

    y_test, X_test, y_train_station, X_train_station = [], [], [], []

    # Split data for each species
    for species in unique_species:
        where_species = species == y_station
        y_species = y_station[where_species]
        X_species = X_station[where_species]
        midpoint = len(y_species) // 2
        
        # Append testing data
        y_test.append(y_species[midpoint:])
        X_test.append(X_species[midpoint:])
        
        # Append training data for 'normal' mode
        y_train_station.append(y_species[:midpoint])
        X_train_station.append(X_species[:midpoint])

    # Concatenate results
    y_test = np.concatenate(y_test)
    X_test = np.concatenate(X_test)
    y_train_station = np.concatenate(y_train_station)
    X_train_station = np.concatenate(X_train_station)

    # Handle different training modes
    if training_mode == 'transfer':
        y_train = labels['species'][~where_station]
        X_train = features[~where_station]
    else:
        y_train = np.concatenate([labels['species'][~where_station], y_train_station])
        X_train = np.concatenate([features[~where_station], X_train_station])

    logging.info(f"Training set: {len(y_train)} samples, Test set: {len(y_test)} samples")
    return X_train, y_train, X_test, y_test


def read_data(max_distance, station, unique_species, training_mode):
    """
    Read and preprocess the data from files.

    Parameters:
    - max_distance: Maximum distance to filter the data.
    - station: The station label to split test and training data.
    - unique_species: List of species to filter and process.
    - training_mode: 'normal' or 'transfer' for how to split training/testing data.

    Returns:
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    """
    feature_file = '../data/scattering_coefficients_32_32.nc'
    logging.info("Starting data reading and preprocessing")
    
    features, labels = read_features_and_labels(feature_file)
    features, labels = filter_species(labels, features, unique_species)
    features, labels = filter_by_distance(features, labels, max_distance)
    
    return split_train_test(features, labels, station, training_mode, unique_species)

def load_model_params(max_distance):
    """
    Load the parameters for the best performing SVM model.

    Parameters:
    - max_distance: Maximum distance used for filtering.

    Returns:
    - svc_params: Dictionary of parameters for the SVM model.
    """

    params_file = f'../data/svc/distance_cases/{str(max_distance).zfill(3)}/best_SVC.txt'
    with open(params_file, 'r') as file:
        data = file.read()

    # Convert string to dictionary
    svc_params = ast.literal_eval(data)
    logging.info(f"Model parameters loaded from {params_file}")
    
    return svc_params


def train_svm_model(X_train, y_train, svc_params):
    """
    Train an SVM model using the provided parameters.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - svc_params: Dictionary of parameters for the SVM model.

    Returns:
    - clf: Trained SVM classifier.
    """
    logging.info("Training SVM model with parameters: %s", svc_params)
    clf = SVC(**svc_params, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    logging.info("SVM model training complete")
    return clf


def evaluate_model(model, X_test, y_test, station, max_dist, unique_species, mode):
    """
    Evaluate the model and visualize the results.

    Parameters:
    - model: Trained model to evaluate.
    - X_test: Test features.
    - y_test: True labels.
    - station: Station label.
    - max_dist: Maximum distance for filtering.
    - unique_species: List of unique species.
    - mode: Training mode used.

    Returns:
    - balanced_accuracy_test: Balanced accuracy score for the test set.
    """
    logging.info("Evaluating model on test data")
    y_pred = model.predict(X_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred)
    logging.info(f"Balanced accuracy: {balanced_accuracy_test:.2f}")

    cm = confusion_matrix(y_test, y_pred, labels=unique_species)
    cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    path2save = f'../data/svc/transfer_cases/{station}_{max_dist}m'
    os.makedirs(path2save, exist_ok=True)

    plot_confusion_matrix(y_test, y_pred, unique_species, path2save)
    save_model_and_data(model, cm_normalized, balanced_accuracy_test, unique_species, station, mode, path2save)

    return balanced_accuracy_test


def plot_confusion_matrix(y_test, y_pred, unique_species, path2save):
    """
    Plot the normalized confusion matrix for the test set.

    Parameters:
    - y_test: True labels for the test set.
    - y_pred: Predicted labels for the test set.
    - station: Station label used.
    - max_dist: Maximum distance for filtering.
    - unique_species: List of species.
    """
    cm = confusion_matrix(y_test, y_pred, labels=unique_species)
    cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm_normalized):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > 0.5 else 'black')

    ax.set_xticks(np.arange(len(unique_species)))
    ax.set_yticks(np.arange(len(unique_species)))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(unique_species)
    ax.set_yticklabels(unique_species)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
    plt.savefig(f'{path2save}/confusion_matrix.png')
    plt.close()


def save_model_and_data(model, cm_normalized, balanced_accuracy_test, unique_species, station, mode, path2save):
    """
    Save the trained model and confusion matrix to disk.

    Parameters:
    - model: Trained SVM model.
    - cm_normalized: Normalized confusion matrix.
    - balanced_accuracy_test: Balanced accuracy score.
    - unique_species: List of species.
    - station: Station label.
    - mode: Training mode used.
    """

    model_file = f'{path2save}/svc_model_{mode}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved as {model_file}")

    data_dict = {
        'confusion_matrix': cm_normalized,
        'balanced_accuracy': balanced_accuracy_test,
        'species': unique_species,
        'station': station
    }

    data_filename = os.path.join(path2save, f'cm_{mode}.pkl')
    with open(data_filename, 'wb') as f:
        pickle.dump(data_dict, f)
    logging.info(f"Confusion matrix data saved as {data_filename}")


# Example usage with hypothetical parameters
if __name__ == "__main__":
    max_distance = 50
    stations = ['ETA00', 'STA02', 'NWP05']
    unique_species = ['elephant', 'giraffe', 'hyena', 'zebra']
    training_modes = ['normal', 'transfer']

    for training_mode in training_modes:
        for station in stations:
            logging.info(f"Starting training for station {station} with mode {training_mode}")
            # Read and preprocess data
            X_train, y_train, X_test, y_test = read_data(max_distance, station, unique_species, training_mode)

            # read svc parameter from best performing model
            svc_params = load_model_params(max_distance)

            # Train SVM model
            model = train_svm_model(X_train, y_train, svc_params)

            # Evaluate model
            evaluate_model(model, X_test, y_test, station, max_distance, unique_species, training_mode)
            logging.info(f"Training for station {station} with mode {training_mode} complete")