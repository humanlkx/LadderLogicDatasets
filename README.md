# YOLO Object Detection Demo

## Installation
To set up the environment, use Conda:
```bash
conda env create -f yolo_environment.yml
```
This will create and configure the necessary environment for running the YOLO-based object detection demo.

## Data and Files
- **`testdata/`**: Contains test files, including:
  - `test.png`: Sample test image
  - `MC7.bin`: Corresponding binary test code for `test.png`
- **`yolodata.7z`**: Contains the dataset used for training/testing
- **`yololabel.txt`**: File containing category labels for YOLO
- **`LD2object.py`**: The demo script for object detection testing

## Usage
To run the demo, execute the following command:
```bash
python LD2object.py
```
Ensure that the dataset and label files are properly extracted and available in the appropriate directories.

## Notes
- Make sure the YOLO dataset is extracted before running the demo.
- If you encounter any missing dependencies, activate the environment using:
  ```bash
  conda activate yolo_environment
  ```
  and install any required packages manually using `pip` or `conda`.
# Related work has been submitted to Tosem
## License
This project is for research and testing purposes only.

