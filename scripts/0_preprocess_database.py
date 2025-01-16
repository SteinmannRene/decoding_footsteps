import os
import numpy as np
import pickle
import xarray as xr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(file_path):
    """Load the dataset from a NetCDF file.

    Args:
        file_path (str): Path to the NetCDF dataset.

    Returns:
        xarray.Dataset: The loaded dataset.
    """
    try:
        dset = xr.open_dataset(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
        return dset
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def replace_station_integers_with_labels(dset, station_dict_path):
    """Replace station integer values in the dataset with station labels.

    Args:
        dset (xarray.Dataset): The dataset containing the 'seis' variable.
        station_dict_path (str): Path to the pickle file containing station label mapping.

    Returns:
        xarray.Dataset: Updated dataset with 'seis' replaced by station labels.
    """
    try:
        station_values = dset['seis'].to_numpy()
        with open(station_dict_path, 'rb') as f:
            station_dict = pickle.load(f)
        logging.info("Station dictionary loaded successfully.")

        station_labels = np.empty(len(station_values), dtype='U10')
        for key, value in station_dict.items():
            station_labels[station_values == value] = key

        dset = dset.drop_vars('seis')
        dset['station'] = xr.DataArray(station_labels, dims='traces')
        logging.info("Station integers replaced with labels.")
        
        return dset
    except FileNotFoundError:
        logging.error(f"Station dictionary file not found: {station_dict_path}")
        raise
    except KeyError:
        logging.error("Mismatch in station dictionary keys and dataset values.")
        raise
    except Exception as e:
        logging.error(f"Error replacing station integers: {e}")
        raise

def replace_integers_with_labels(dset, species_names, family_names):
    """Replace species integer values in the dataset with species labels.

    Args:
        dset (xarray.Dataset): The dataset containing the 'class' variable.
        species_names (list): List of species names corresponding to class values.

    Returns:
        xarray.Dataset: Updated dataset with 'class' replaced by species labels.
    """
    try:
        species_values = dset['class'].to_numpy()
        unique_values = np.unique(species_values)

        if len(species_names) != len(unique_values) and len(family_names) != len(unique_values):
            raise ValueError("Number of labels must match the number of unique values.")
        
        species_dict = dict(zip(species_names, unique_values))
        family_dict = dict(zip(family_names, unique_values))
        species_labels = np.empty(len(species_values), dtype='U10')
        family_labels = np.empty(len(species_values), dtype='U14')

        for key, value in species_dict.items():
            species_labels[species_values == value] = key

        for key, value in family_dict.items():
            family_labels[species_values == value] = key

        dset = dset.drop_vars('class')
        dset['species'] = xr.DataArray(species_labels, dims='traces')
        dset['family'] = xr.DataArray(family_labels, dims='traces')
        logging.info("Species integers replaced with labels.")
        
        return dset
    except ValueError as ve:
        logging.error(f"Species label error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error replacing species integers: {e}")
        raise

def update_dataset_with_images_and_energy(dset):
    """Update the dataset with correctly dimensioned image data and signal energy.

    Args:
        dset (xarray.Dataset): The original dataset containing 'image' and 'chunk' variables.

    Returns:
        xarray.Dataset: Updated dataset with new 'image' data array and 'signal_energy' variable.
    """
    try:
        images = xr.DataArray(
            dset['image'].data, 
            dims=["traces", "side1", "side2", "comp"]
        )

        dset = dset.drop_vars('image').assign(image=images)
        dset = dset.assign_coords(comp=['vertical', 'north', 'east'])

        signal_energy = (np.abs(dset['chunk'])**2).sum(dim='time')
        dset = dset.assign(signal_energy=signal_energy)
        
        logging.info("Dataset updated with new image dimensions and signal energy.")
        return dset
    except KeyError as ke:
        logging.error(f"Missing variable in dataset: {ke}")
        raise
    except Exception as e:
        logging.error(f"Error updating dataset: {e}")
        raise

def delete_species_and_station(dset, species_to_delete, station_to_delete):
    """Delete specified species and station records from the dataset.

    Args:
        dset (xarray.Dataset): The original dataset containing 'species' and 'station' variables.
        species_to_delete (str): Species to be removed from the dataset.
        station_to_delete (str): Station to be removed from the dataset.

    Returns:
        xarray.Dataset: Updated dataset with the specified species and station records removed.
    """
    try:
        mask_station = dset['station'] != station_to_delete
        mask_species = dset['species'] != species_to_delete
        dset = dset.sel(traces=mask_station & mask_species)
        
        logging.info(f"Records for species '{species_to_delete}' and station '{station_to_delete}' deleted.")
        return dset
    except KeyError as ke:
        logging.error(f"Variable not found in dataset: {ke}")
        raise
    except Exception as e:
        logging.error(f"Error deleting species or station: {e}")
        raise

def save_dataset(dset, file_path):
    """Save the updated dataset to a NetCDF file.

    Args:
        dset (xarray.Dataset): The dataset to be saved.
        file_path (str): Path to save the NetCDF file.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Existing file at {file_path} removed.")

        dset.to_netcdf(file_path)
        logging.info(f"Dataset saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")
        raise

def main():
    """Main function to process the dataset with various transformations."""
    try:
        # Define paths
        dataset_path = '../data/original/all_species/dset_allspec_150_chunks_clean.nc'
        station_dict_path = '../data/original/all_species/seis_dict.pkl'
        save_path = '../data/modified_dataset.nc'
        
        # Species names for replacement
        species_names = [
            'human', 'elephant', 'warthog', 'leopard', 'guineafowl',
            'dikdik', 'giraffe', 'hyena', 'rabbit', 'zebra', 'hippo'
        ]

        family_names = [
            'HOMINIDAE', 'ELEPHANTIDAE', 'SUIDAE', 'FELIDAE', 'NUMIDIDAE',
            'BOVIDAE', 'GIRAFFIDAE', 'HYAENIDAE', 'LEPORIDAE', 'EQUIDAE', 'HIPPOPOTAMIDAE'
        ]

        # Load the dataset
        dset = load_dataset(dataset_path)

        # Replace station and species integers with labels
        dset = replace_station_integers_with_labels(dset, station_dict_path)
        dset = replace_integers_with_labels(dset, species_names, family_names)

        # Update dataset with images and signal energy
        dset = update_dataset_with_images_and_energy(dset)

        # Delete records for specific species and station
        dset = delete_species_and_station(dset, species_to_delete='dikdik', station_to_delete='ETC00')

        # Save the modified dataset
        save_dataset(dset, save_path)
        logging.info("Dataset processing complete.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    main()
