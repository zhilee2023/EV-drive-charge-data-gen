# EV-drive-charge-data-gen
This repository is an optimized version of the article "Synthetic Data Generation for Joint Electric Vehicle Driving and Charging Events via Deep Generative Networks." It implements deep generative networks for synthesizing joint electric vehicle driving and charging event data with various improvements over the original approach.

### 📦 Dependencies

This project relies on the following core Python packages:

- **[PyTorch 1.11.0](https://pytorch.org/)** – For deep learning and model development, including training and inference.
- **[Pandas 2.2.3](https://pandas.pydata.org/)** – For efficient data manipulation, preprocessing, and analysis of large tabular datasets.
- **[NumPy 1.24.4](https://numpy.org/)** – For fast numerical computations, array handling, and mathematical operations.


### 📁 Generated Data

Due to the sensitive nature of the original dataset, we are unable to release it publicly.  Instead, we provide a synthetic dataset containing approximately **25,600 vehicles** and **60 drive/charge events per vehicle** in the file `sample.csv`.

This sample data can be used for tasks such as **model training**, **fine-tuning**, or **algorithm evaluation**. **Please note:** since the synthetic data differs from the original dataset in certain statistical and temporal characteristics, training results may also vary accordingly.

### 🚀 Usage Overview

This project provides three main scripts for model usage: `train.py`, `fine_tune.py`, and `sample.py`.  
Each script requires a configuration file (`config.json`) and, if applicable, a path to a pre-trained model.

We have uploaded a pre-trained model `models_test.pth`, which can be used directly for sample generation or fine-tuning.  
By default, the sampling process generates **60 drive/charge events for a single vehicle**, but users can customize the number of events by modifying the `max_event_len` parameter in `sample.py`.

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


### 🎲 3. Generate Synthetic Data
Once a model is trained or fine-tuned, you can generate synthetic driving and charging events by running:

```python
python sample.py --config config.json --models.test.pth
```
The generated output will be saved in the format (e.g., CSV).


Pre-trained Model
For quick testing and sample generation, we provide a ready-to-use pre-trained model named models_test.pth. To use this model with sample.py or fine_tune.py, update your config.json with the following settings:

json

{
  "d_model": 128,
  "layers": 48,
  "model_type": "TransformerGibbs"
}
Then, run either:

```python
python sample.py --config config.json --model_path models_test.pth
```

```python
python fine_tune.py --config config.json --model_path models_test.pth
```
This allows you to generate synthetic data directly or fine-tune the pre-trained model on your own dataset.

