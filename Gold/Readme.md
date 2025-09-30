# Practical Blueprint: QLoRA Fine-Tuning Mistral-7B-Instruct-v0.2 for user assistance in C & Linux System Programming
## What you get
1. This project fine-tunes a Mistral-7B-Instruct-v0.2 instruct model using QLoRA on a T4 (16GB) GPU.
2. Supervised fine tune a model based on User instruction and assistance response using TRL SFTT and PEFT QLora training on a small dataset related with Linux system programming (you can replace with your own)
3. Domain-focused formatting for C & Linux system programming
4. A synthetic dataset generator (sysprog_dataset) that generates multiple problems/solutions in the area of system programming
5. Model is trained(fine tuned) on instruction and response format prompt for the generated synthetic data 
6. Safe defaults for google colab free tier T4 GPU: 4-bit quantization, small per-device batch, gradient accumulation
7. Quick evaluation and inference by applying Lora weights on base model.


## For real training, plug in your larger dataset and extend training time/epochs.
