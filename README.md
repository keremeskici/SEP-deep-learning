# Emotion Recognition Project

This project implements a deep learning model to recognize emotions from facial expressions. We use a modified ResNet18 architecture trained on the RAF-DB dataset.

## Setup

We used **Python 3.11.14** for this project. To run the code, you first need to install the required libraries.

You can install them with pip:

pip install -r requirements.txt

### Device Support
The code automatically detects your hardware. It supports:
- **Mac (Apple Silicon)**: Uses MPS (Metal Performance Shaders) for GPU acceleration.
- **NVIDIA GPUs**: Uses CUDA.
- **CPU**: Fallback if no GPU is found.

## Data Preparation

The dataset used is RAF-DB (Real-world Affective Faces Database). Due to the file size limit (50MB), the dataset is not included in this submission. You need to download it separately.

1. Create a folder named "datasets" in the main project directory.
2. Inside "datasets", create a folder named "RAF-DB".
3. Download the RAF-DB dataset and extract it.
4. Place the images and label CSV files so the structure looks like this:

datasets/
  RAF-DB/
    DATASET/
      train/
        1/
          train_00001.jpg
          ...
      test/
        ...
    train_labels.csv
    test_labels.csv

The code uses "datasets/RAF-DB/DATASET" as the root for images and looks for the CSV files in "datasets/RAF-DB/".

## Project Architecture

We chose a modular approach with multiple scripts for single operations. This structure helped us in two main ways:
1. **Team Collaboration**: Multiple folders and scripts allowed us to distribute tasks among team members easily (e.g., one person on training, another on the demo).
2. **Debugging and Reusability**: Using specific scripts for operations made debugging more efficient. We could also reuse components (like the model builder or transforms) across different scripts clearly.

## How to Run

### Training

To train the model, run the training script. It uses the configuration from "configs/default_config.yaml".

python scripts/train.py

**Outputs**:
- A folder named `outputs` will be created. All different training runs are saved here (models, checkpoints).
- We use **Weights & Biases (wandb)** for metric tracking. Make sure to login to wandb if you want to track experiments.

### Webcam Demo

You can run a realtime demo using your webcam. You need to provide the path to a trained model checkpoint (file ending in .pth).

python scripts/realtime_webcam.py --model_path outputs/models/run_XX/checkpoints/best_model.pth

Press 'q' to quit the webcam window.

### Batch Demo

If you have a folder of images you want to classify, use the demo script. It runs the model on all images in the folder and saves the results to "outputs/predictions.csv".

python scripts/demo.py path_to_your_image_folder --model_path outputs/models/run_XX/checkpoints/best_model.pth

### Video Demo

You can process a video file to classify emotions frame-by-frame. The script will output a new video overlaid with emotion classifications, confidence levels, and saliency maps (Grad-CAM) showing the important regions. It generates the output video directly inside the `outputs` directory.

python scripts/video_demo.py --video_path inputs/my_video.mp4 --model_path outputs/models/run_XX/checkpoints/best_model.pth

## Code Structure

- **configs/**: Contains the configuration file for training.
- **data/**: Scripts for loading and transforming data.
- **models/**: Defines the deep learning model architecture (Modified ResNet18).
- **scripts/**: Main executable scripts for training and demos.
- **utils/**: Helper functions for checkpoints, metrics, and visualization.
