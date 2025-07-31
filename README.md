# Flood Segmentation from Satellite Imagery

A simple pipeline for semantic segmentation of flooded regions using Sentinel satellite data and deep learning.

## Project Overview
This repository implements a U-Net based model to automatically detect and segment flood extents in multispectral Sentinel-2 imagery. Flood mapping is critical for rapid response in disaster management, and automating this process helps scale analysis over large areas.

## Dataset
We use the “Flood Detection Using Sentinel Satellite Pictures” dataset from Kaggle:
- **Source:** https://www.kaggle.com/code/shaunakmandal/flood-detection-using-sentinel-satellite-pictures  
- **Composition:**  
  - **Images:** Sentinel-2 MSI (RGB + NIR) patches covering flood-prone areas  
  - **Masks:** Binary flood labels derived from SAR-based inundation maps  

## Preprocessing
Before training, each image–mask pair goes through:
1. **Loading & Decoding**  
 - Read image bytes (`tf.io.read_file`) and decode as 3-channel JPEG (`tf.image.decode_jpeg(..., channels=3)`).  
 - Read mask bytes and decode as single‐channel JPEG (`channels=1`).

2. **Resizing**  
 - Resize both to `128×128` pixels.  
 - Use **bilinear** interpolation for images, and **nearest‐neighbor** (`method="nearest"`) for masks to preserve binary labels.

3. **Normalization & Casting**  
 - Cast image to `tf.float32` and scale to [0,1] by dividing by 255.  
 - Cast mask to `tf.float32`, divide by 255, **round** to {0,1}, then cast to `tf.int32` to ensure a binary mask.

4. **tf.data Pipeline**  
 ```python
 ds = tf.data.Dataset.from_tensor_slices((images, masks))
 ds = (ds
       .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
       .cache()
       .batch(16)
       .prefetch(tf.data.AUTOTUNE))
```
## Models Used

We experimented with the following model architectures for flood segmentation:

### U-Net
- Baseline encoder–decoder structure with skip connections  
- Efficient for biomedical and satellite image segmentation  
- Output activation: `sigmoid` for binary mask prediction  

### U-Net + Attention
- Adds **attention gates** to skip connections  
- Helps the model focus on relevant spatial features (e.g., flooded areas)  
- Reduces noise from irrelevant background features  

### U-Net + Inception Modules
- Replaces standard convolutional blocks with **Inception modules** to capture multi-scale context  
- Enables the model to detect both small and large flood zones in the same frame  

### U-Net + Attention + Inception
- Combines both **Attention Gates** and **Inception Blocks**  
- Improves spatial precision and multi-scale feature extraction  

Each of these variants was evaluated under identical training conditions to ensure a fair comparison.

