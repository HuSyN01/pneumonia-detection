# Chest X-Ray Pneumonia Classifier

## Overview
This project is a beginner-friendly implementation of a binary image classification model to detect pneumonia from chest X-ray images using deep learning. It uses the **MobileNetV2** pre-trained model with TensorFlow/Keras to classify X-rays as **Normal** (healthy) or **Pneumonia** (infected). The dataset is sourced from Kaggle’s "Chest X-Ray Images (Pneumonia)" dataset.

This project is ideal for those new to AI and medical imaging, demonstrating transfer learning and image preprocessing. **Note**: This is for educational purposes only and not for clinical use.

## Dataset
The dataset is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, containing ~5,800 chest X-ray images split into:
- **Train**: ~5,216 images (Normal and Pneumonia).
- **Validation**: 16 images.
- **Test**: 624 images.

Images are labeled as `NORMAL` or `PNEUMONIA` in subfolders.

## Prerequisites
- **Python 3.7+**
- **Libraries**:
  - TensorFlow (`tensorflow>=2.0`)
  - NumPy
  - Matplotlib
  - Pandas
  - Scikit-learn
- **Optional**: Google Colab for free GPU access (recommended for beginners).
- **Dataset**: Download from Kaggle (see Setup).

## Setup Instructions
1. **Download the Dataset**:
   - Go to [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and download the dataset (~1GB).
   - Unzip it to get the `chest_xray` folder with `train`, `val`, and `test` subfolders.
   - Alternatively, use the Kaggle API in Colab:
     ```bash
     !pip install kaggle
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
     !unzip chest-xray-pneumonia.zip
     ```
     - Get `kaggle.json` from your Kaggle account (Profile > Account > Create New API Token).

2. **Install Dependencies**:
   In your Python environment or Colab:
   ```bash
   pip install tensorflow keras numpy matplotlib pandas scikit-learn
   ```

3. **Directory Structure**:
   Ensure your project folder looks like:
   ```
   chest_xray_project/
   ├── chest_xray/
   │   ├── train/
   │   │   ├── NORMAL/
   │   │   ├── PNEUMONIA/
   │   ├── val/
   │   ├── test/
   ├── chest_xray_classifier.py
   ├── README.md
   ```

## Usage
1. **Run the Script**:
   - Update the `train_dir`, `val_dir`, and `test_dir` paths in `chest_xray_classifier.py` to match your dataset location.
   - Run the script:
     ```bash
     python chest_xray_classifier.py
     ```
   - In Colab, copy the script into a cell and run it.

2. **What It Does**:
   - Loads and preprocesses images (resized to 150x150, normalized).
   - Displays a sample X-ray image with its label.
   - Trains a MobileNetV2-based model for 3 epochs (adjustable).
   - Evaluates accuracy on the test set (~80-90% expected).
   - Predicts on a single test image (e.g., `chest_xray/test/NORMAL/IM-0001-0001.jpeg`).

3. **Output**:
   - A plot of a sample X-ray.
   - Training progress (accuracy/loss per epoch).
   - Test accuracy.
   - Prediction for a single image (Normal or Pneumonia).

## Notes
- **Memory**: The dataset is large. Use a smaller `batch_size` (e.g., 16) or `target_size` (e.g., 100x100) if you face memory issues.
- **Colab**: Enable GPU (Runtime > Change runtime type > GPU) for faster training.
- **Improvements**:
  - Increase `epochs` (e.g., 10-20) for better accuracy.
  - Add data augmentation (e.g., `rotation_range=20` in `ImageDataGenerator`).
  - Unfreeze some MobileNetV2 layers for fine-tuning.

## Troubleshooting
- **FileNotFoundError**: Check dataset paths in the script.
- **Memory Errors**: Reduce `batch_size` or image size, or use Colab’s GPU.
- **Low Accuracy**: Increase `epochs` or add augmentation.
- **Kaggle API Issues**: Re-download `kaggle.json` from Kaggle.

## Future Work
- Try multi-class classification (e.g., bacterial vs. viral pneumonia).
- Explore other datasets (e.g., brain MRI for tumors).
- Deploy the model as a web app using Flask or Streamlit.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Framework: TensorFlow/Keras
- Inspiration: Beginner-friendly AI for medical imaging
