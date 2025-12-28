# Environmental Sound Classifier with Explainable AI for Enhanced Recognition

## Overview

This application utilizes a **lightweight yet high-performing CNN model - CNN-PSK**, enhanced with advanced techniques like **PCA pooling**, **Sparse Salient Region Pooling (SSRP)**, and **Kolmogorov-Arnold Networks (KAN)**, to classify environmental sounds. The model processes uploaded audio clips and returns:

- **Class Category**: The predicted category the sound belongs to.
- **Label Confidence**: The confidence score for the predicted label.

Additionally, the application provides **explainability** features to help users understand the model's decision-making process, including visualizations and detailed frequency and temporal explanations.

## Features

### 1. **Sound Classification**

- **Class Category**: The uploaded audio clip is analyzed, and the model predicts which class/category the sound belongs to.
- **Label Confidence**: The confidence score for the predicted class/category is provided alongside the label.

### 2. **Explainability Features**

The application provides multiple layers of explainability through visualizations and frequency/temporal analysis:

- **Spectrogram of the Original Sound**: A graphical representation of the audio signal.
  
- **Grad-CAM Heatmap**: A visualization that highlights the areas of the input that the model focused on when making the classification decision.

- **Grad-CAM Heatmap with Bounding Boxes**: A more detailed Grad-CAM visualization that includes bounding boxes to emphasize the critical areas of the sound.

- **Spectrogram Overlaid with Bounding Boxes**: The spectrogram with bounding boxes drawn over the most critical time and frequency segments that contributed to the classification.

#### Results Preview for Clock-tick Sound

| Spectrogram | Grad-CAM |
|------------|----------|
| ![](/backend/example_visualizations/clock_tick/clock_tick_spectrogram.png) | ![](/backend/example_visualizations/clock_tick/clock_tick_gradcam.png) |

| Spectrogram overlaid with Bounding Boxes | Grad-CAM Heatmap with Bounding Boxes |
|------------------|---------------|
| ![](/backend/example_visualizations/clock_tick/clock_tick_spec_with_bboxes.png) | ![](/backend/example_visualizations/clock_tick/clock_tick_gradcam_with_bboxes.png) |

### 3. **Frequency Explanations**

Based on the Grad-CAM results, the application provides frequency-related explanations, indicating whether the sound exhibits any of the following characteristics:

- **Mostly High Frequencies**
- **Mostly Low Frequencies**
- **Mostly Mid Frequencies**
- **Low-Mid Frequencies**
- **Mid-High Frequencies**
- **Broadband**: Identifying peak frequency styles

### 4. **Temporal Explanations**

The application also provides temporal explanations by analyzing the model's focus over time:

- **Burst**: Sounds with short, rapid bursts of activity.
- **Repetitive**: Sounds with repetitive patterns over time.
- **Continuous**: Sounds that maintain a continuous activity over time.
- **Intermittent**: Sounds that occur at irregular intervals.

### 5. **Time Segments Extraction**

The model identifies the most important time segments within the uploaded audio. These time segments represent the portions of the sound that the model focused on most during classification. Users can play and listen to these extracted time segments to better understand the classification decision.

  