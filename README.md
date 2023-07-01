# visiope
Vison and Perception Project - Sapienza (AIRO) 2022/2023

# Mars Terrain Semantic Segmentation

This project aims to solve a semantic segmentation problem over a dataset of Mars terrain images collected by three NASA rover missions. The objective is to build a lightweight neural network that improves autonomous driving for future rover missions by recognizing different kinds of terrains. The task involves segmenting the images into four classes plus background.

## Directory Structure

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

## Getting Started

To get started with the project, please follow the steps below:

1. Clone the repository: `git clone https://github.com/your-username/mars-terrain-segmentation.git`
2. Set up the dataset by running the data loading scripts in the `dataloader/` directory.
3. Explore the different models available in the `models/` directory and choose the one you want to use.
4. Train the chosen model using the training code provided.
5. Evaluate the trained model using the evaluation code provided.
6. Experiment with different loss functions from the `loss/` directory if desired.
7. Customize the code as per your requirements and explore other functionalities.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code according to your needs.


