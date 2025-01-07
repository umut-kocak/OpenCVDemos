# OpenCVDemos

# Repository Overview

This repository contains a collection of Python-based OpenCV demos.

## Running the Demos

To run these demos successfully, follow these steps:

- **Create the Conda Environment**: Use the `requirements_conda` file to create a corresponding Conda environment.

  ```bash
  conda env create -f ./requirements_conda.yaml --prefix ./env
  conda activate ./env
  ```

- **Run the Demo in Module Mode**: After setting up the environment, run the demo from the root directory in module mode. For example:


  ```bash
  python -m demos.face_detection.face_detection
  ```
Ensure you are in the root directory before executing the Python command to run each demo.
