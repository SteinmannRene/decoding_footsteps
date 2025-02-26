# Decoding the Footsteps of the African Savanna: Classifying Wildlife Using Seismic Signals and Machine Learning

### Overview
This repository provides the codes for reproducing the results and figures of the paper.

### Table of Contents
1. [Installation](#installation)
2. [Download data](#download)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

### Installation
1. **Clone the repository or download the zip**:

2. **Set up the environment**:
   - Using Conda:
     ```bash
     conda env create -f environment.yml
     conda activate your-env-name
     ```

### Download data

The original data can be found at [dropbox](https://www.dropbox.com/sh/p1swf94hs2pa47g/AACWTAXGlgrjc1GtOaNKURCFa?dl=0), published along with the [paper](https://doi.org/10.1002/rse2.242). Please download the folder "dset_allspec_150" and unpack its content in a path called "data/original/all_species/" within the repository. If you save it at a different place, you need to change the paths within the codes to load the data.

### Usage
The folder scripts contains the Python scripts to process the data and create new data products. The folder notebooks contains jupyter notebooks to produce the figures. First, run the scripts according to the numbering and then run the notebooks according to the numbering.

### Project Structure

```bash
decoding_footsteps/
│
├── data/                   # Folder for storing datasets and results (not included in the repository)
│
├── notebooks/              # contains notebooks to create figures and results
│
├── scripts/                # contains scripts to preprocess dataset, calculate scattering coefficient, etc.
│
├── environment.yml         # Conda environment configuration
├── README.md               # This readme file
└── license.txt             # License file
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions or support, please contact:

- Name: Rene Steinmann
- Email: rene.steinmann@gfz.de
- GitHub: [@SteinmannRene](https://github.com/SteinmannRene)

---
