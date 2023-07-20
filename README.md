# Mars Terrain Semantic Segmentation
Vison and Perception Project - Sapienza (AIRO) 2022/2023

---
## Table of content
0. [Introduction](#introduction)
1. [Getting Started](#getting-started)
2. [Train the model](#train-the-model)
3. [Test the model](#test-the-model)
4. [Content of this repository](#content-of-this-repository)
---

# Introduction
This project aims to solve a semantic segmentation problem over a dataset of Mars terrain images collected by three NASA rover missions. The objective is to build a lightweight neural network that improves autonomous driving for future rover missions by recognizing different kinds of terrains. The task involves segmenting the images into four classes plus background.

# Getting Started

To get started with the project, please follow the steps below:

- If you want to try it on your local maachine:

    1. Install the requirment in requirment.txt
    2. Clone the repository: `git clone https://github.com/your-username/mars-terrain-segmentation.git`
    3. Set up the dataset by running the data loading scripts in the `datasetup/` directory.
    4. Or just run one of the scripts inside the `experiment` directory.
    5. Explore the different models available in the `models/` directory and choose the one you want to use.
       
- If you wnat to try it on [Google COLAB](https://colab.research.google.com/):
    1. Just open in COLAB one of the scripts in the `experiment` directory.


# Train the model

# Test the model

# Content of this repository

The project directory has the following structure:

- `dataloader/`: Contains the data loading code.
    - `load.ipynb`: Jupyter Notebook for loading and preprocessing the dataset.
    - `load.py`: Python script for loading and preprocessing the dataset.

- `gitsetup.ipynb`: Jupyter Notebook providing instructions for setting up Git.

- `latex/`: Contains LaTeX files for any documentation or reports related to the project.
    - `visiope.zip`: Compressed file containing LaTeX files for visualization and reporting.

- `loss/`: Contains custom loss functions for the semantic segmentation task.
    - `custom_loss.py`: Implementation of custom loss functions.
    - `__init__.py`: Python package file.
    - `__pycache__/`: Directory containing compiled Python bytecode files.
        - `custom_loss.cpython-310.pyc`: Compiled bytecode file.
        - `loss_functions.cpython-310.pyc`: Compiled bytecode file.
        - `trainer_module.cpython-310.pyc`: Compiled bytecode file.
    - `test/`: Contains test files for the loss functions.
        - `fake_prediction_lable.pt`: Example fake prediction label file.
        - `test_loss.py`: Unit tests for the loss functions.
        - `test_trainer.py`: Unit tests for the trainer module.
        - `true_lable.pt`: Example true label file.
    - `trainer_module.py`: Implementation of a trainer module.

- `models/`: Contains the neural network models for the semantic segmentation task.
    - `base_model.py`: Base model implementation.
    - `heads/`: Contains custom head modules for the models.
        - `custom.py`: Custom head module implementation.
    - `lawin.py`: Implementation of the Lawin model.
    - `main.ipynb`: Jupyter Notebook for model training and evaluation.
    - `resDecode.py`: Implementation of the ResDecode model.
    - `segFormerpp.py`: Implementation of the SegFormer++ model.
    - `segformer.py`: Implementation of the SegFormer model.
    - `Training.ipynb`: Jupyter Notebook for model training.

- `project-managing/`: Contains project management-related files.
    - `abstarct.pdf`: Abstract document describing the project.
    - `AI4Mars-AI4Space-final.pdf`: Final project report.
    - `ProjectSheet.pdf`: Project sheet outlining the project details.
    - `RESOURCES.md`: Markdown file listing additional resources related to the project.

- `test.ipynb`: Jupyter Notebook for testing various components of the project.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code according to your needs.


