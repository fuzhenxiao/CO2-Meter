# CO2-Meter (AAAI 2026)

This repository contains the official implementation for the paper **"CO2-Meter"**, accepted to AAAI 2026.
It includes raw data, data preprocessing, feature extraction, model training, and evaluation scripts.

## Repository Structure

```
fuzhenxiao/
│
├── llm_config/           # Configuration files for large language models
├── output/               # Training and evaluation outputs (generated automatically)
├── raw data/             # Contains 40k raw data samples (requires decompression)
│
├── build_dataset.py      # Build PyTorch datasets (.pt files)
├── dataset.py            # Dataset class
├── extract_feature.py    # Feature extraction to build graphs
├── hardware.py           # Hardware configurations
├── make_graph.py         # build graphs
├── model.py              # Energy-Prediction model definition
└── train_and_test.py     # Training and testing pipeline
```

## Instructions

### 1. Prepare Raw Data

The `raw data` folder contains approximately 40,000 raw data samples.
You need to **decompress** the data before use.
Depending on your training configuration, you need to **split the raw data** into multiple CSV files.

### 2. Build the Dataset

Use the following command to generate `.pt` files for PyTorch from CSV files:

```
python build_dataset.py
```

**Note:** `build_dataset.py` requires a **Hugging Face token** to extract parameters from a language model.
You can obtain a token from your Hugging Face account

### 3. Train and Test the Model

After preparing the dataset, configure which .pt files to use and run the training script:

```
python train_and_test.py
```

During training, you should see logs similar to:

```
2025-11-10 18:26:50,926 - train_and_test.py:241 - INFO -
Epoch[29/30](6750/8000) Lr:0.00100000 Loss:0.25221 MAPE:0.34330 ErrBnd(0.1):0.20634,(0.05):0.09999,(0.01):0.02311
```

This indicates that training is running correctly.
The model checkpoints will be saved as `.pth` files.
