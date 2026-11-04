#  Waste-Sorting: AI-Powered Waste Classification

An AI-powered waste classification system that uses deep learning and computer vision to identify different types of waste from images. The project aims to support smarter waste segregation by automatically classifying waste into categories that can help improve recycling and waste management.

The system allows users to upload waste images through a web application, receive a predicted category, view the model's confidence, and explore predictions across different waste categories.

---



##  Project Overview

Waste management is an important environmental challenge. Improper waste segregation can reduce recycling efficiency, increase landfill waste, and make waste processing more difficult.

Traditional waste segregation often depends on manual identification and sorting. This can be time-consuming and inconsistent.

This project explores how deep learning and computer vision can be used to automate waste classification.

The system takes an image as input and predicts the category of waste using a trained convolutional neural network.

The project also includes a Streamlit-based web application that allows users to interact with the trained model without requiring knowledge of Python or machine learning.

The long-term goal is to develop a waste classification system that can be integrated with automated waste segregation hardware.

**Current implementation:** Image-based waste classification through a web application.

**Future direction:** Real-time camera classification and automated physical waste sorting.

---

##  Problem Statement

Develop an image classification system that can identify different categories of waste using deep learning.

The system should:

1. Accept an image of waste as input.
2. Process the image using appropriate preprocessing techniques.
3. Predict the waste category using a trained deep learning model.
4. Display the predicted category and confidence score.
5. Show the probability distribution across the supported categories.
6. Provide examples of images belonging to different waste categories.

The project serves as a prototype for an AI-powered waste segregation system.

---

##  Objectives

- Develop a deep learning model for waste classification.
- Experiment with pretrained convolutional neural networks.
- Compare different model architectures.
- Evaluate classification performance using appropriate metrics.
- Build an interactive web application using Streamlit.
- Explore the possibility of integrating the classifier with automated waste sorting hardware.

---

##  Features

### 1. Image-Based Waste Classification

Upload an image of waste and receive a predicted category from the trained model.

### 2. Confidence Score

Display the model's predicted probability for the selected category.

The confidence score represents the model's probability estimate and does not guarantee that the prediction is correct.

### 3. Probability Distribution

View the model's predictions across all supported waste categories, ranked from highest to lowest probability.

### 4. Interactive Web Interface

Use a Streamlit application to upload images and visualize predictions.

### 5. Deep Learning Experiments

Train and evaluate multiple pretrained architectures to compare their performance on waste classification.

---

##  Supported Waste Categories

The current application uses the following 10 categories:

| Category | Description |
|---|---|
| Biological | Biodegradable or biological waste |
| Battery | Battery waste |
| Cardboard | Cardboard packaging and boxes |
| Clothes | Clothing and textile waste |
| Glass | Glass bottles and glass materials |
| Metal | Metal waste and containers |
| Paper | Paper waste and paper products |
| Plastic | Plastic containers and packaging |
| Shoes | Footwear waste |
| Trash | General miscellaneous waste |

**Note:** The supported categories depend on the dataset and model checkpoint used by the application. The model can only classify categories that were included during training.

---

##  Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model training and inference |
| Torchvision | Pretrained models and image preprocessing |
| ResNet50 | Image classification architecture |
| YOLO Classification | Additional model experiment |
| Streamlit | Interactive web application |
| Pillow | Image loading and processing |
| NumPy | Numerical operations |
| Pandas | Dataset organization and analysis |
| Scikit-learn | Dataset splitting and evaluation metrics |
| Matplotlib and Seaborn| Visualization and evaluation plots |
| Git and GitHub | Version control and project collaboration |

---

##  Project Workflow

The project follows the workflow below:

```text
                  WASTE IMAGE DATASET
                          |
                          v
                DATASET EXPLORATION
                          |
                          v
               IMAGE PREPROCESSING
                          |
                          v
                TRAIN / VALIDATION /
                    TEST SPLIT
                          |
                          v
                PRETRAINED MODEL
                  FINE-TUNING
                          |
                          v
                 MODEL EVALUATION
                          |
                          v
                SAVE BEST CHECKPOINT
                          |
                          v
                 STREAMLIT WEB APP
                          |
                          v
                 UPLOAD WASTE IMAGE
                          |
                          v
                 IMAGE PREPROCESSING
                          |
                          v
                MODEL CLASSIFICATION
                          |
                          v
               PREDICTED CATEGORY
                 + CONFIDENCE SCORE
                          |
                          v
               DISPLAY PROBABILITIES
```

---

##  Dataset

### RealWaste Dataset

The initial experiments used the RealWaste image classification dataset.

Datasets link:

https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/
https://www.kaggle.com/datasets/mostafaabla/garbage-classification


The datasets contains images belonging to different waste categories and is suitable for image classification experiments.

The dataset was organized into class-specific folders.

### Dataset Organization

The classification dataset follows a folder-based structure:

```text
Data2/
└── original/
    ├── battery/
    ├── biological/
    ├── cardboard/
    ├── clothes/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── shoes/
    └── trash/
```

Each image belongs to a category based on the folder in which it is stored.

### Dataset Preparation

The dataset preparation process included:

1. Loading images from the class folders.
2. Checking image dimensions and file formats.
3. Checking for corrupted images.
4. Exploring the number of images in each category.
5. Creating stratified training, validation, and testing splits.
6. Applying image preprocessing and augmentation.
7. Preparing PyTorch datasets and dataloaders.

The initial RealWaste dataset contained **27773 images across 10 categories**.

The dataset was split into:

| Split | Images |
|---|---:|
| Training | 19441 |
| Validation | 4166 |
| Testing | 4166 |
| Total | 27773 |

Stratified splitting was used to maintain approximately the same class distribution across the splits.

### Image Preprocessing

The image preprocessing pipeline includes:

- Resizing images to the input dimensions required by the pretrained model.
- Converting images into PyTorch tensors.
- Normalizing images using ImageNet statistics when using ImageNet pretrained weights.

Training transformations may also include data augmentation to improve generalization.

---

##  Model Development

The project explores transfer learning using pretrained deep learning architectures.

Transfer learning allows a model pretrained on a large image dataset to be adapted to a smaller waste classification dataset.

Instead of training the entire network from scratch, pretrained weights are used as a starting point.

### 1. MobileNetV3 Small

MobileNetV3 Small was used as an initial baseline model.

The pretrained feature extractor was adapted to classify the waste categories.

The experiment included:

- Loading pretrained ImageNet weights.
- Replacing the final classification layer.
- Training the classifier.
- Fine-tuning selected feature layers.
- Evaluating the model using validation data.
- Saving the best model checkpoint.

MobileNetV3 was selected as a lightweight architecture suitable for exploring classification performance and potential deployment constraints.

### 2. ResNet50

ResNet50 was also explored using pretrained ImageNet weights.

The final classification layer was replaced to match the number of waste categories.

The model was trained and evaluated using the prepared dataset.

ResNet50 was used to investigate whether a deeper architecture could improve classification performance.

The trained checkpoint used by the web application is:

```text
best_waste_resnet50_weighted.pth
```

The checkpoint contains the trained model parameters and the class mapping required for inference.


---

##  Model Evaluation

The models are evaluated using classification metrics to understand their performance across different waste categories.

### Evaluation Metrics

**Accuracy**

Measures the proportion of correctly classified images.

**Precision**

Measures how many images predicted as a particular category actually belong to that category.

**Recall**

Measures how many images belonging to a category were correctly identified.

**F1 Score**

Combines precision and recall into a single metric.

**Confusion Matrix**

Shows the relationship between actual categories and predicted categories.

### Model Comparison

The experiments use the same dataset split wherever applicable.

The comparison considers:

- Validation accuracy.
- Macro-averaged F1 score.
- Per-class precision.
- Per-class recall.
- Confusion matrix.
- Training behavior.

Macro F1 is particularly useful because the dataset contains different numbers of images per category.

### ResNet50 Test Results

The trained ResNet50 model achieved the following reported test performance:

| Metric | Result |
|---|---:|
| Test Accuracy | 98.32% |
| Macro Precision | 97.96% |
| Macro Recall | 98.07% |
| Macro F1 Score | 98.01% |

These results represent performance on the prepared test split and should not be interpreted as guaranteed performance on real-world camera images.

The model may perform differently on images with different backgrounds, lighting conditions, camera angles, or waste materials.

---

##  Web Application

The project includes a Streamlit application for interactive waste classification.

The application loads the trained ResNet50 checkpoint and allows users to upload waste images.

### Application Features

#### Image Upload

Users can upload images in the following formats:

- JPG
- JPEG
- PNG
- WEBP

#### Prediction

The uploaded image is passed through the trained ResNet50 model.

The application displays:

- Uploaded image.
- Predicted waste category.
- Model confidence.
- Probability distribution across all categories.

#### Category Examples

The application also displays example images from the dataset for the predicted categories.

This allows users to compare their uploaded image with examples from the training dataset.

### Application Design

The application uses a dark interface with teal accents and a waste-classification theme.

The interface includes:

- Category sidebar.
- Image upload section.
- Image analysis section.
- Predicted category display.
- Confidence visualization.
- Category probability display.
- Dataset example images.
  

---


##  Limitations

The current project is a prototype and has several limitations.

### 1. Image-Level Classification

The model predicts one category for an entire image.

It does not identify multiple waste objects in the same image.

### 2. Dataset Dependence

The model's performance depends on the dataset used for training.

Images with unfamiliar backgrounds, lighting conditions, camera angles, or waste materials may produce incorrect predictions.

### 3. Confidence Is Not a Guarantee

A high confidence score does not necessarily mean that the prediction is correct.

The model may confidently misclassify an image.

### 4. Dataset Category Limitations

The model can only predict categories included in its training dataset.

It does not currently provide a dedicated hazardous-waste detection system.

### 5. No Physical Sorting Hardware

The current application performs image classification only.

It does not control motors, robotic arms, conveyor belts, or physical waste bins.

### 6. Real-World Validation

The reported test performance is based on the prepared dataset split.

Additional testing using images captured from real collection points is necessary before considering deployment.

---

##  Future Improvements

The project can be extended in several directions.

### Model Improvements

- Experiment with additional pretrained architectures.
- Compare model size and inference speed.
- Improve performance on difficult waste categories.
- Investigate class imbalance.
- Evaluate model calibration and confidence reliability.

### Computer Vision Improvements

- Implement object detection for multiple waste items.
- Annotate images with bounding boxes.
- Detect multiple waste objects in a single camera frame.
- Investigate segmentation for more precise object identification.

### Application Improvements

- Add webcam-based classification.
- Add prediction history.
- Add image preprocessing options.
- Improve the visualization of model predictions.
- Add support for selecting different trained models.

### Hardware Integration

- Integrate a camera with a waste collection system.
- Connect classification predictions to a microcontroller.
- Control a motorized sorting mechanism.
- Develop a prototype using a conveyor belt or rotating sorting platform.
- Implement a manual-review or unknown-category pathway for uncertain predictions.

### Real-World Deployment

- Collect images from actual waste collection environments.
- Evaluate performance under different lighting conditions.
- Test different camera positions.
- Investigate model deployment on edge devices.
- Measure inference latency and hardware requirements.

---

##  Conclusion

Waste-Sorting explores the use of deep learning and computer vision for automated waste classification.

The project includes dataset preparation, transfer learning, model evaluation, experimentation with different classification architectures, and an interactive Streamlit application.

The current implementation demonstrates how a trained image classification model can identify waste categories from uploaded images.

The project provides a foundation for future development toward real-time waste classification and automated waste segregation systems.

---

##  Contributors

**PMSAK**

GitHub:

https://github.com/PMSAK

Project Repository:

https://github.com/PMSAK/Waste-Sorting

---


---

⭐ If you find this project interesting, consider starring the repository!

