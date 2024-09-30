# Decoding footsteps

### Overview
This project uses machine learning to classify wildlife species based on seismogram data. It processes seismographic signals and applies a Support Vector Machine (SVM) model to detect footfalls of different species.

### Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

### Features
- **Species Classification**: Detect and classify different wildlife species using seismographic signals.
- **Distance-Based Filtering**: Filter signals by maximum distance and species before model training.
- **Station-Specific Models**: Train models separately for each station in the provided list of stations.

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up the environment**:
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Using Conda:
     ```bash
     conda env create -f environment.yml
     conda activate your-env-name
     ```

### Usage
1. **Preprocess the Data**: 
   - Edit the parameters in the configuration file: `config.py`.
   - Run the preprocessing script:
     ```bash
     python preprocess_data.py
     ```

2. **Train the Model**:
   - Train the SVM model for multiple stations:
     ```bash
     python train_model.py --stations 'ETA00, STA02, NWP05' --max_distance 50
     ```

3. **Evaluate the Model**:
   - Evaluate the trained model for a specific station:
     ```bash
     python evaluate_model.py --stations 'NWP05' --max_distance 50
     ```

### Project Structure

```bash
your-repo-name/
│
├── data/                   # Folder for storing datasets and results
│   ├── raw/                # Raw data (input files)
│   └── processed/          # Preprocessed data
│
├── models/                 # Trained models (saved during training)
│
├── src/                    # Source code folder
│   ├── preprocess_data.py  # Script to preprocess the data
│   ├── train_model.py      # Script to train the SVM models
│   ├── evaluate_model.py   # Script to evaluate the models and generate confusion matrices
│   └── utils.py            # Utility functions (e.g., reading data, filtering)
│
├── requirements.txt        # Pip requirements file
├── environment.yml         # Conda environment configuration
├── README.md               # This readme file
└── LICENSE                 # License file
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

You can adjust the sections like the project name, repository URL, or the contact details as needed.
