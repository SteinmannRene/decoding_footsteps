# Decoding the Footsteps of the African Savanna: Classifying Wildlife Using Seismic Signals and Machine Learning

### Overview
This repository provides the codes for reproducing the results and figures of the paper.

### Table of Contents
1. [Installation](#installation)
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

### Usage
The folder scripts contains the Python scripts to process the data and create new data products. The folder notebooks contains jupyter notebooks to produce the figures. First, run the scripts according to the numbering and then run the notebooks according to the numbering.

### Project Structure

```bash
decoding_footsteps/
│
├── data/                   # Folder for storing datasets and results (not included in the repository)
│
├── notebooks/                 # Trained models (saved during training)
│
├── scripts/                    # Source code folder
│   ├── preprocess_data.py  # Script to preprocess the data
│   ├── train_model.py      # Script to train the SVM models
│   ├── evaluate_model.py   # Script to evaluate the models and generate confusion matrices
│   └── utils.py            # Utility functions (e.g., reading data, filtering)
│
├── environment.yml         # Conda environment configuration
├── README.md               # This readme file
└── license.txt             # License file
```

### Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add a feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions or support, please contact:

- Name: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---
