import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainerCallback, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datetime import datetime
from pathlib import Path
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import keyboard
import ast
import pickle
import time
import math
import joblib
import os
import glob

# Reduce on plateau learning rate scheduler
class MyReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, patience=3, factor=0.5, min_lr=1e-7):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.scheduler = None

    def on_train_epoch_begin(self, args, state, control, **kwargs):
        # Initialize the scheduler with the optimizer
        optimizer = kwargs.get('model', None).optimizer
        if optimizer:

            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=args.optimizers[0],
                mode='min',
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
                verbose=True
            )
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Monitor validation loss and reduce learning rate if needed
        eval_loss = kwargs.get("eval_loss", None)
        if eval_loss is not None:
            self.scheduler.step(eval_loss)

# Warmup with reduce on plateau learning rate scheduler
class WarmupReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, optimizer, num_warmup_steps, patience=3, factor=0.5):
        self.num_warmup_steps = num_warmup_steps
        self.linear_warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / float(self.num_warmup_steps)),
        )
        self.plateau = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        self.step_count = 0
        self.in_warmup = True

    def on_step_begin(self, args, state, control, **kwargs):
        if self.step_count < self.num_warmup_steps:
            self.linear_warmup.step()
        self.step_count += 1

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.step_count >= self.num_warmup_steps and "eval_loss" in metrics:
            self.plateau.step(metrics["eval_loss"])
        return control

# Cosine annealing learning rate scheduler
class CosineAnnealingWithDecay(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.9, warmup_steps=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.warmup_steps = warmup_steps

        self.T_i = T_0
        self.cycle = 0
        self.max_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Warmup phase
        if step < self.warmup_steps:
            warmup_factor = float(step) / float(max(1, self.warmup_steps))
            return [lr * warmup_factor for lr in self.max_lrs]

        # Adjusted step after warmup
        adjusted_step = step - self.warmup_steps

        # Determine current cycle
        T_i = self.T_i
        cycle = 0
        while adjusted_step >= T_i:
            adjusted_step -= T_i
            T_i = T_i * self.T_mult
            cycle += 1

        # Decay the base LR for the current cycle
        current_max_lrs = [lr * (self.decay_factor ** cycle) for lr in self.max_lrs]

        # Cosine annealing formula
        return [
            self.eta_min + (max_lr - self.eta_min) * (1 + math.cos(math.pi * adjusted_step / T_i)) / 2
            for max_lr in current_max_lrs
        ]

# Custom trainer for cosine annealing
class CustomSFTTrainer(SFTTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer = AdamW(self.model.parameters(), lr=2e-3) #2e-3

        self.lr_scheduler = CosineAnnealingWithDecay(
            optimizer=self.optimizer,
            T_0=1800,               # number of steps in the first cycle - 1000
            T_mult=1,              # cycle length multiplier - 1
            eta_min=1e-6,          # minimum LR - 1e-6
            decay_factor=0.6,      # max LR decay between cycles - 0.6
            warmup_steps=200  #200     # warmup steps - 100
        )

        self._lr_scheduler = self.lr_scheduler  # make it accessible for trainer internals

# Format data for input to model
def format_data(sample, dataset_directory):

    user_content = []

    sample["Images"] = ast.literal_eval(sample["Images"])

    for img in sample["Images"]:
        user_content.append(
            {"type": "image", 
             "image": dataset_directory + img,
            })
        
    user_content.append({"type": "text", "text": "Describe this:"})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert numismatist that accurately describes items based on visual information."}],
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["Description"]}]
        }
        ]

# Custom data collator
def collate_fn(examples):
    """
    Custom data collator for Vision-Language Model (VLM) fine-tuning.
    Takes a list of raw dataset examples, processes the text and images,
    and formats them into padded PyTorch tensors ready for the model.
    """

    # Prepare texts for processing
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # Process the images to extract inputs
    image_inputs = [process_vision_info(example)[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    image_tokens = [151652, 151653, 151655]

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

# Preprocess or load dataset
def preprocessing(format_dataset, dataset_directory):
    # Load, format and split dataset
    if format_dataset:

        dataset = load_dataset('csv', data_files=str(Path(dataset_directory+"numismatic_image_description.csv").resolve()))
        dataset = dataset["train"].shuffle(seed=42)

        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
        temp_split = dataset_split["test"].train_test_split(test_size=0.02, seed=42)

        train_dataset = dataset_split["train"]
        val_dataset = temp_split["train"]
        test_dataset = temp_split["test"]

        train_dataset = [format_data(sample, dataset_directory) for sample in train_dataset]
        val_dataset = [format_data(sample, dataset_directory) for sample in val_dataset]
        test_dataset = [format_data(sample, dataset_directory) for sample in test_dataset]

        # Save
        with open(dataset_directory + "../pickled_dataset/train.pkl", "wb") as f:
            pickle.dump(train_dataset, f)
        with open(dataset_directory + "../pickled_dataset/val.pkl", "wb") as f:
            pickle.dump(val_dataset, f)
        with open(dataset_directory + "../pickled_dataset/test.pkl", "wb") as f:
            pickle.dump(test_dataset, f)

    else:

        # Load
        with open(dataset_directory + "../pickled_dataset/train.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open(dataset_directory + "../pickled_dataset/val.pkl", "rb") as f:
            val_dataset = pickle.load(f)
        with open(dataset_directory + "../pickled_dataset/test.pkl", "rb") as f:
            test_dataset = pickle.load(f)

    return train_dataset, val_dataset, test_dataset

# Runs inference on one sample
def generate_text_from_sample(model, processor, messages, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(messages)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])


if __name__ == '__main__':
    # Is the model finetuned and ready for inference
    finetuned = False

    # Do you need to format dataset
    format_dataset = True

    # Enter dataset directory
    dataset_directory = "Path/to/dataset"

    # Load the finetuned model
    if finetuned:
        # Path to model checkpoint
        model_id = "Path/to/model/checkpoint"

        # Load finetuned model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load default processer
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=334*28*28)

        # Preprocess dataset
        train_dataset, val_dataset, test_dataset = preprocessing(format_dataset, dataset_directory)

        # Test the model inference
        for test in test_dataset:

            for img_block in test[1]["content"][0:-1]:
                img = Image.open(img_block["image"])
                img.show()

            start = time.time()
            print(test[:2])
            generate_text_from_sample(model, processor, test[:2])
            end = time.time()

            print(test[2]["content"][0]["text"])

            print("The time of execution of above program is :", (end-start) * 10**3, "ms")

            keyboard.wait('space')

    # Load the default model from Hugging Face
    else:
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        # BitsAndBytesConfig int-4 config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load default model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config
        )

        # Load default processer
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=334*28*28)

        # Preprocess dataset
        train_dataset, val_dataset, test_dataset = preprocessing(format_dataset, dataset_directory)

        # Low-Rank Adaptation (LoRA) Configuration
        peft_config = LoraConfig(
        lora_alpha=32, # Scaling factor for the weight updates
        lora_dropout=0.05,
        r=16, # Rank of the update matrices
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Layers in the transformer architecture to apply LoRA
        task_type="CAUSAL_LM",
        )

        # Configure training arguments
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = SFTConfig(
            output_dir=f"./Qwen2.5-VL-3B-Intruct-Numismatic/checkpoints/{run_name}",  # Directory to save the model
            logging_dir=f"./Qwen2.5-VL-3B-Intruct-Numismatic/logs/{run_name}", # Directory for logging tensorboard
            num_train_epochs=6, #8 # Number of training epochs
            per_device_train_batch_size=1,  # Batch size for training
            per_device_eval_batch_size=1,  # Batch size for evaluation
            gradient_accumulation_steps=32,  # Steps to accumulate gradients
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            label_names=["labels"], # Set label names used in collator
            
            # Optimizer and scheduler settings
            optim="adamw_torch_fused",  # Optimizer type
            learning_rate=2e-4,  # Learning rate for training
            lr_scheduler_type="linear",  # Type of learning rate scheduler
            #lr_scheduler_kwargs={
            #"num_cycles": 2  # Set the number of cosine cycles
            #},
            #dataloader_num_workers=4,

            # Logging and evaluation
            logging_steps=10,  # Steps interval for logging
            eval_steps=200,  # Steps interval for evaluation
            eval_strategy="steps",  # Strategy for evaluation
            save_strategy="steps",  # Strategy for saving the model
            save_steps=200,  # Steps interval for saving
            metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            greater_is_better=False,  # Whether higher metric values are better
            load_best_model_at_end=True,  # Load the best model after training

            # Mixed precision and gradient settings
            bf16=True,  # Use bfloat16 precision
            tf32=True,  # Use TensorFloat-32 precision
            max_grad_norm=1.0,  # Maximum norm for gradient clipping
            warmup_ratio=0.01,  # Ratio of total steps for warmup

            # Hub and reporting
            push_to_hub=False,  # Whether to push model to Hugging Face Hub
            report_to="tensorboard",  # Reporting tool for tracking metrics

            # Gradient checkpointing settings
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing

            # Dataset configuration
            dataset_text_field="",  # Text field in dataset
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            remove_unused_columns = False,  # Keep unused columns in dataset
            max_length=1024  # Maximum sequence length for input
        )

        # Run this in terminal to see tensorboard: tensorboard --logdir=./Qwen2.5-VL-3B-Intruct-Numismatic/logs

        # Initialize the trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            peft_config=peft_config,
            processing_class=processor,

            # optimizers=(optimizer, None),
            # callbacks=[WarmupReduceLROnPlateauCallback(optimizer, num_warmup_steps=warmup_steps, patience=3, factor=0.5)]

            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5),
                MyReduceLROnPlateauCallback(patience=3, factor=0.5, min_lr=1e-7)
            ]

        )

        # Train the model
        trainer.train()


        # Save final trained model
        trainer.save_model(training_args.output_dir)


        # Save best model and its processor
        model.save_pretrained("best_finetuned_model_linear_full_data")
        processor.save_pretrained("best_finetuned_model_linear_full_data")


        # Save the merged model and its processor
        merged_model = model.merge_and_unload()

        # If the model is quantized, dequantize the new merged model
        if hasattr(merged_model, "dequantize"):
            merged_model = merged_model.dequantize()

        # Save the actual merged model and its processor
        merged_model.save_pretrained("best_finetuned_model_linear_full_data_merged")
        processor.save_pretrained("best_finetuned_model_linear_full_data_merged")
