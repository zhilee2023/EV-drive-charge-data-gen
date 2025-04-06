# EV-drive-charge-data-gen
This repository is an optimized version of the article "Synthetic Data Generation for Joint Electric Vehicle Driving and Charging Events via Deep Generative Networks." It implements deep generative networks for synthesizing joint electric vehicle driving and charging event data with various improvements over the original approach.

### 📦 Dependencies

This project relies on the following core Python packages:

- **[PyTorch 1.11.0](https://pytorch.org/)** – For deep learning and model development, including training and inference.
- **[Pandas 2.2.3](https://pandas.pydata.org/)** – For efficient data manipulation, preprocessing, and analysis of large tabular datasets.
- **[NumPy 1.24.4](https://numpy.org/)** – For fast numerical computations, array handling, and mathematical operations.


### 📁 Generated Data

Due to the sensitive nature of the original dataset, we are unable to release it publicly.  Instead, we provide a synthetic dataset containing approximately **25,600 vehicles** and **60 drive/charge events per vehicle** in the file `sample.csv`.

This sample data can be used for tasks such as **model training**, **fine-tuning**, or **algorithm evaluation**.

### 🚀 Usage Overview

This project provides three main scripts for model usage: `train.py`, `fine_tune.py`, and `sample.py`.  
Each script requires a configuration file (`config.json`) and, if applicable, a pre-trained model path.

---

### 🏋️‍♂️ 1. Training from Scratch
To train a new model using your own dataset, simply run:

```python
python train.py --config config.json
```


### 🔧 2. Fine-tune a Pre-trained Model
To fine-tune a pre-trained model on your specific dataset, execute:

```python
python fine_tune.py --config config.json --models.test.pth
```
This script loads an existing pre-trained model and continues training using the settings specified in config.json.
Before fine-tuning, consider adjusting parameters such as learning rate, batch size, and dataset paths to match your new data and fine-tuning objectives.


