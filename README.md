
# Impact of Training Data Quality on YOLOv8n Performance for Crack Detection in Concrete Structures

## Introduction

This project evaluates the performance of the YOLOv8n model in detecting cracks in concrete structures, focusing on the impact of training data quality. The evaluation compares models trained on clean datasets with those trained on datasets augmented with noise and blur, and tests these models under various image quality conditions.

The `Original results` folder contains the results of the two notebooks, including:
- Training results (weights, curves, metrics)
- Validation results for both clean and degraded training methods
- CSV files with performance metrics (recall, precision)

## Citation

If you use this project or parts of it, please cite the following reference for the project:

```bibtex
@misc{
    weisberg2024impact,
    title={Impact of Training Data Quality on YOLOv8n Performance for Crack Detection in Concrete Structures},
    author={Ran Weisberg and Artemiy Tumanov},
    year={2024},
    howpublished={\url{https://github.com/yourusername/your-repo-name}},
}
```

## Prerequisites

To run the code, you will need to install the following libraries:

- ultralytics
- numpy
- opencv-python
- pillow
- pyyaml
- matplotlib
- jupyter

You can install these libraries using pip:

```bash
pip install ultralytics numpy opencv-python pillow pyyaml matplotlib jupyter
```

## Run the Test

To run the tests, follow these steps:

1. **Clean Dataset**:
   Open `Train_and_Test_clean_dataset.ipynb` and run all cells. This notebook will train a YOLOv8n model for 50 epochs on the clean dataset and evaluate it on various test sets.

2. **Degraded Dataset**:
   Open `Train_and_Test_degraded_dataset.ipynb` and run all cells. This notebook will train a YOLOv8n model for 50 epochs on the degraded dataset and evaluate it on various test sets.

## Modify and Run

### Generating New Test Sets

To generate new test sets with different levels of noise and blur:

1. Open `generate_degraded_testsets.py`.
2. The script works with user input to generate new sets. When you run the script, it will prompt you to enter the mode (noise/blur) and the number of sets. For each set, you will be prompted to enter the percentage of noise or blur.
3. Run the script:
   ```bash
   python generate_degraded_testsets.py
   ```
4. The output will be folders of the new sets in the `test_datasets` directory.

### Including New Test Sets in the Notebooks

To include new test sets in the notebooks:

1. Add the new test set directories under `test_datasets`.
2. Update the `data_sets` list in the notebooks (`Train_and_Test_clean_dataset.ipynb` and `Train_and_Test_degraded_dataset.ipynb`) to include the paths to the new test sets.
3. Run the notebooks as usual to train and evaluate the models with the new test sets.

### Changing the Degraded Train Set Amount of Noise and Blur

1. Open `generate_degraded_datasets.py`.
2. Adjust the `percentage` parameter in the `add_gaussian_noise` and `add_gaussian_blur` functions to the desired level of noise and blur.
3. Run the script:
   ```bash
   python generate_degraded_datasets.py
   ```

## Acknowledgment

The clean training data used in this project, DawgSurfaceCracks, was downloaded from Roboflow and is not owned by this project. Please acknowledge the original dataset as follows:

```bibtex
@misc{
    dawgsurfacecracks_dataset,
    title = { DawgSurfaceCracks Dataset },
    type = { Open Source Dataset },
    author = { XplodingdoG },
    howpublished = { \url{ https://universe.roboflow.com/xplodingdog/dawgsurfacecracks } },
    url = { https://universe.roboflow.com/xplodingdog/dawgsurfacecracks },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { may },
    note = { visited on 2024-07-03 },
}
```

The YOLOv8 model used in this project is provided by Ultralytics. Please acknowledge the YOLOv8 model as follows:

```bibtex
@misc{jocher2023yolov8,
  title={YOLOv8: You Only Look Once version 8},
  author={Jocher, G. and Chaurasia, A. and Qiu, J. and LeGrand, A.},
  year={2023},
  howpublished={\url{https://github.com/ultralytics/yolov8}},
  note={Accessed: 2024-06-08}
}
```
