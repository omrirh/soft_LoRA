# Soft LoRA
### This repo suggests re-production steps of LoRA performance experiment.
### It uses GLUE SST2 task for fine-tuning BERT on downstream tasks and testing their results.

## Pre-requisites:
- Make sure your machine is CUDA compatible
- `docker` is installed

## Running the experiment containerized
```bash
make setup-run
```

#### After the training & evaluation phases are completed, you should see the session artifacts under `results` in your project.

