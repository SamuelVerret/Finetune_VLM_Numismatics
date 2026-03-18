# Qwen2.5-VL Numismatic Item Description

A vision-language model (VLM) fine-tuning pipeline using Qwen2.5-VL-3B to automatically generate professional, highly accurate auction descriptions from images of rare coins, banknotes, jewelry, gold, and silver. 

## 🔒 Private Dataset

The dataset used to train this model contains proprietary images and expert descriptions. To protect this competitive advantage, the raw data is strictly private and not included. The code provided demonstrates the formatting, QLoRA fine-tuning, and inference logic.

## ⚙️ Installation

Ensure you have a CUDA-compatible GPU, then install the necessary dependencies:

`pip install -r requirements.txt`

## 🚀 Usage

Execution is controlled by two boolean flags at the bottom of the main script.

### 1. Training
Place your dataset in the specified directory and set the flags to process the data and begin fine-tuning:

`finetuned = False`
`format_dataset = True # Set to False after the first run to reuse .pkl files`

### 2. Tensorboard
Monitor training logs in a separate terminal with: `tensorboard --logdir=./logs`*

### 3. Inference & Testing
Once training is complete, update the `model_id` path to your saved checkpoint and set the flags to test mode:

`finetuned = True`
`format_dataset = False`

The script will iterate through your test set, display the item image, print the generated auction description, and wait for you to press `Space` before moving to the next item.