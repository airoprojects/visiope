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

    - Install the requirment in requirment.txt
    - Clone the repository: `git clone https://github.com/your-username/mars-terrain-segmentation.git`
    - Set up the dataset by running the data loading scripts in the `datasetup/` directory.
    - Or just run one of the notebboks inside the `experiment` directory.
    - Explore the different models available in the `models/` directory and choose the one you want to use.
       
- If you wnat to try it on a shared machine in Google colab
    - Open [Google colab](https://colab.research.google.com/) and import one of the notebooks `experiment`.
    - Or just click on the colab link in one of the scripts inside the `experiment` directory.


# Train the model

You can use our custom modules such as `/tools/traner` and `/tools/loss` to train different models in the main nootebook underr `experiments`.
There you chan find our custom models: **SEGFORMER-PP** and **SEGNET-PP**. You can also train State of the art models by running one of the `test-models` notebooks

# Test the model

To test the models you ca as well use our custom tester inside `/tools/tester`. In there you can find a class responsable to test the accuracy of different models and some functions dedicated to produce segmenation images taken from a test subset of the original dataset.

# Content of this repository

The project directory has the following structure:

- `datasetup/`: Contains the data loading code.
    - `datasetup.ipynb`: Jupyter Notebook for loading and preprocessing the dataset.
    - `datasetup.py`: Python script for loading and preprocessing the dataset.
      
- `expriments`: Contains four jupiter nootebooks to make experiments and try out different models.
  
- `Tools/`: Contains custom loss functions for the semantic segmentation task.
    - `data/`: Data module.
        - `utils.py` : Contains classes to download, pre-process and load the dataset.
    - `loss/`: Loss Module.
        - `loss.py` : Contains classes that implements different loss functions.
    - `tester/`: Test module
        - `tester.py`: Contains a class to test different models.
    - `trainer/`: Train Module
        - `trainer.py`: Contains a class to perform different kind of training over different models.

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

- `resources`: Contains project management-related files.
    - `abstarct.pdf`: Abstract document describing the project.
    - `AI4Mars-AI4Space-final.pdf`: Final project report.
    - `ProjectSheet.pdf`: Project sheet outlining the project details.
    - `resources.md`: Markdown file listing additional resources related to the project.
 
- `gitsetup.ipynb`: Jupyter Notebook providing instructions for setting up Git on colab.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code according to your needs.


