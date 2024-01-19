# Data Processing
The process.py script is tailored for processing the house numbers dataset sourced from Stanford University. This dataset comprises images of house numbers captured from Google Street View.

### Configurations:
- USE_SAMPLE_DATA: If set to True, the script will process a sample of the dataset. For a comprehensive analysis, set it to False to leverage the full dataset. To give you a perspective on size, the full dataset boasts 531,131 images, whereas the sample dataset holds 26,032.
- SAMPLE_SIZE: When opting for the sample data by setting USE_SAMPLE_DATA to True, you can dictate the exact sample size to further refine the dataset's scope. If you've chosen to process the full dataset, this option becomes non-operational.

### Functionality Overview:
- The script first checks if the data has been downloaded. If not, it proceeds to fetch the data from the designated URL.
- Once downloaded, it extracts data from the .mat file.
- Following this, it distributes the data into various subsets: demo, train, validate, and unlabeled, as per the provided ratios.

### Output Structure:
The processed data is structured meticulously. Both the image files and their corresponding labels are saved in specific directories:
- demo: The set of unlabeled images reserved for demostration.
- train: The set of images allocated for training.
- validate: The set of images intended for validation.
- unlabeled: The set of images that lack labels. It is used to mimick active learning scenarios.

Each of the aforementioned directories houses an images folder with the actual image files. In addition, a mappings.csv file is provided, offering a handy reference that maps each image filename to its actual label.

### Running the Script:
1. Ensure any required libraries or dependencies are installed via `pip install -r requirements.txt`
2. Ensure you are in the root folder which is `numbers_captcha`.
3. Before running the script, ensure you've set `USE_SAMPLE_DATA` and `SAMPLE_SIZE` configurations as desired.
4. Execute the Script: `python -m captcha_server.src.preprocessing.process`.
