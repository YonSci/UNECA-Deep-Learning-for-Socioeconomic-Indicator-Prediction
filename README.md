# Unveiling the Potential of Transfer Learning in Machine Learning and Deep Learning Modes for Socioeconomic Indicators Prediction (Consumption Expenditure) in Malawi, 2016

# Incorporating Survey Information into Satellite Imagery

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/intro_im.png)   

## Problem❓

1) The lack of reliable and continuous socioeconomic data in developing countries is a major obstacle to monitoring and evaluating sustainable development goals and making informed policy decisions. Obtaining frequent and reliable national-level statistics through surveys is both costly and labor-intensive, which poses a significant challenge for governmental and non-governmental organizations. 

2) In recent years, there has been a growing interest in utilizing machine learning and deep learning techniques to estimate socioeconomic indicators. However, one of the main challenges associated with these approaches is the lack of reproducibility and documentation, making it difficult to verify and implement them effectively. This makes it difficult to replicate the results and adapt them to different contexts. To address this issue, it is crucial to focus on enhancing the reproducibility of machine learning projects.

By addressing these challenges, organizations and researchers can develop more accurate, reliable, and reproducible machine-learning models for estimating socioeconomic indicators in developing countries. These models can provide valuable insights into the socioeconomic situation in these countries, helping policymakers and organizations make informed decisions and plan effective interventions.

## Solution 💡

1) In response to the first challenge, we present a cost-effective, accurate, and scalable method for predicting socioeconomic indicators. In this project, a novel machine learning/deep learning approach is implemented to predict socioeconomic indicators from survey data, and publicly available nighttime and high-resolution daytime satellite imagery. The methodology was originally established by researchers at [Stanford University.]( https://sustain.stanford.edu/predicting-poverty) and is currently being adopted by several institutions around the world.
  
2) In response to the second challenge,  it is crucial to focus on enhancing the reproducibility of machine learning projects. This can be achieved by ensuring that the code, data, and environment used in the development of the models are well-documented and easily reproducible. By doing so, researchers can not only verify the accuracy and reliability of the models but also adapt them to different contexts and situations, thereby increasing their practical utility and impact. We demonstrated a reproducible approach that provided a step-by-step guideline for data collection, preprocessing, and code implementation.
 
3) Furthermore, we address challenges related to nighttime satellite imagery retrieval by deploying GEE Javascript and resolving the time mismatches with survey data. Additionally, we tackle issues linked to Planet satellite imagery by updating the Planet downloader scripts to accommodate changes in the Planet Data API parameters, ensuring accurate data retrieval. Furthermore, the implementation of advanced deep-learning models allows us to assess the performance of alternative models, enhancing the robustness of the methodology.

## Objective 🎯

The primary aim of this project is to integrate **survey data**,  **nighttime satellite imagery**, and  **daytime high-resolution satellite imagery**, leveraging **machine learning/deep Learning** methodologies to forecast **socioeconomic indicators** using  a reproducible framework utilizing open-source tools and resources (i.e. Consumption expenditure for Malawi for the year 2016).​

### Specific Objectives

- Fill both temporal and spatial socioeconomic data gaps by inferring from remote sensing data. ​

- Enhance the existing methodology using only open-source tools and publicly available data/resources.​

- Improve the reproducibility and documentation of the existing methodology mainly key stages such as data collection, code implementation, and data analysis for easy knowledge/skill transfer.  

## Application 💻

- This methodology holds broad applicability, extending beyond consumption expenditure prediction, it can be applied to wealth, poverty, income, and population prediction.
- It facilitates predictions during "off-years" when surveys are not conducted.
- It also enables near real-time monitoring serving as an early-warning system.
- This kind of approach is scalable, the trained model, from one location, can be applied to new regions with similar characteristics.
- The proposed method contributes to the production of frequent and continuous statistical reports on socioeconomic indicators, complementing existing methods used in the National Statistics Offices (NSOs).


## Implementation Frameworks and Environment 🖥️ 

We used **Google Colab Pro+** for computing with high-performance GPUs—specifically A100 and V100 with 51.0GB of RAM. 

The project utilizes the following framework & tools: ​

1) **PyTorch framework** using Python programming language: PyTorch is a popular deep learning framework that supports Python, making it easier for developers to work with machine learning models and data processing tasks.
   
2) **Google Earth Engine (GEE)** with JavaScript codes: GEE is a cloud-based geospatial data processing platform that allows users to visualize, analyze, and model Earth science data. JavaScript codes can be used to create custom GEE applications and widgets for interactive mapping and data visualization.
  
3) **QGIS**: QGIS is a free and open-source geographic information system (GIS) that allows users to create, analyze, manage, and visualize spatial or geographic data. QGIS can be used for various applications, including mapping, spatial analysis, and visualization.

## High-level Workflow 
​
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/workflow.png)   

## Procedure 📋

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/method1.png)   

The implementation of this approach includes important steps such as:

### 1)  Survey data collection, and processing

The project used publicly available survey data from the [World Bank Living Standards Measurement Study (LSMS)](https://microdata.worldbank.org/index.php/catalog/lsms/) Microdata Library, particularly the [Fourth Integrated Household Survey (IHS4)](https://microdata.worldbank.org/index.php/catalog/2936/data-dictionary/F98?file_name=HouseholdGeovariablesIHS4), gathered through the National Statistical Office (NSO) of Malawi, during the period spanning from April 2016 to April 2017.

Once, the [Survey Data ](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/tree/main/Survey_Data) is retrieved and undergoes a data cleaning process to handle missing values. Subsequently, the consumption values were standardized using [Purchasing Power Parity (PPP)](https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=MW) for Malawi in 2016, and the standardized consumption and Geovariable datasets were merged using their unique ID. The data is then grouped by the enumeration area using coordinates (latitude & longitude), and the daily consumption per person is computed ($/person/day). Following this, statistical information is summarized, and the results are verified with external [World Bank benchmarks](https://data.worldbank.org/indicator/SI.SPR.PCAP?end=2016&locations=MW&start=2016&view=bar​). Finally, the processed data is used to generate consumption maps, providing a visual representation of the predicted socioeconomic indicators. 

For a detailed implementation procedure, go through: [Survey_Data_Preprocessing_Malawi_2016.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Survey_Data_Preprocessing_Malawi_2016.ipynb)

#### Steps: 

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/survey2.png)   

#### Daily Consumption per Capita for Malawi for 2016

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Daily_Per_Capita_Consumption_Malawi_Year_2016.png)   

#### Basic summary statistics​ & distributions of consumption per capita per day 

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/summry_stat1.png)   


### 2) Nightlight satellite imagery acquisition, and processing

The project also utilized nighttime satellite imagery sourced from the [NOAA National Center for Environmental Information](https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html​) and downloaded using the Google Earth Engine (GEE) Javascript code editor. Here you can find the Javascript [code](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/GEE_Nightlight_2016_2017.js).

In the script, we filtered the nighttime satellite imagery using the area of interest (AOI), temporal duration, and the relevant spectral bands required for the analysis.  Then we computed the annual composite of the filtered images. The resulting nightlight image was exported as a GeoTIFF file to Google Drive. You can find the annual composite nighttime satellite imagery for the year 2016 [here](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Nighttime_Satellite_Imagery/malawi_nightlight_2016_viirs.tif). Subsequently, calculations were performed to create a 10kmx10km box around the central latitude and longitude to retrieve the nightlight values for each cluster. The procedure also involved the computation of summary statistics for the nightlight values and calculating their correlation with the consumption value. The processed data is utilized to create maps depicting nightlight values.

The process also included computing summary statistics of the nightlight values and calculating their correlation with the consumption value. Finally, the processed data is used to generate nightlight value maps.

For a detailed implementation procedure, go through [Processing_Nighttime_Satellite_Imagery.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Processing_Nighttime_Satellite_Imagery.ipynb)

#### Steps: 
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/n4.png)

#### Description of Nightlight satellite imagery
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/n1.png)

#### Nightlight satellite imagery 

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/nightlight.png)  

#### Cluster Boxes
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/n2.png)  

#### Map of Nightlight Values and Consumption for Malawi for 2016
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/nightlight_consumtion.png) 

#### Correlation between  Nightlight values and Consumption 
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/n3.png) 

### 3) Generate download locations for Daytime satellite imagery 

We created download locations for daytime satellite images using a combination of **systematic** and **stochastic** sampling methods within cluster bounding boxes. Specifically, we generated 50 download locations per cluster, forming a grid of 49 uniformly spaced points (7x7) within the bounding box and adding 1 point through random sampling within the same box. This approach ensures diverse download locations for daytime satellite images. Subsequently, for each set of 50 points, we compiled the image name, image latitude, and image longitude, appending them to the data frame. A total of **39,000** image download locations have been generated. 

For a detailed implementation procedure, go through [Generate_Image_Download_Locations.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Generate_Image_Download_Locations.ipynb)

#### Steps:
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/download_location.png) 

#### Download Location Dataframe
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/download_location4.png) 

#### Download Location Map
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/generate_down_loc.png) 

### 4) Undersamping to avoid nightlight values class bias​ 
Within our dataset, the number of nightlight values having zero is notably higher, potentially causing an imbalance in the data distribution. The objective is to address this by undersampling or reducing instances from areas with zero or minimal nightlight data, aiming to mitigate class imbalance. This approach introduces diversity into the model by selectively removing rows associated with zero nightlights until the target fraction is achieved. A total of **33,900** image download locations have been left after performing the under-sampling.

For a detailed implementation of the undersampling, go through [Generate_Image_Download_Locations.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Generate_Image_Download_Locations.ipynb) 

#### Steps:

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/down_sampling.png)  

 #### Download Location Undersamped Dataframe 
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/download_location3.png) 

### 5) Daytime satellite imagery acquisition and processing

The acquisition of high-resolution daytime satellite imagery is performed using the Planet API, which provides images specifically for research and academic purposes. The **Planet Scope (PSScene)** images have a spatial resolution ranging from 3.7 to 4.1 meters, later resampled to **3 meters** for practical use. The process of obtaining Planet Imagery encompasses a series of steps. Initially, we set up the API Key in Planet Explorer. Following this, we apply essential filters such as geometry, date, and cloud filters to download the images properly. The download locations (image latitude and longitude) derived from previous steps serve as inputs for image retrieval, incorporating additional parameters like a zoom level of 14 and a maximum cloud filter of 0.05 (5%). The image acquisition spans the period from 2016 to 2017, culminating in a total of **33,900** downloaded images.

#### Steps:

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/download_planet.png)   

For downloading the Planet daytime satellite imagery, go through [Download_satellite_images_Planet.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Download_satellite_images_Planet.ipynb) 

#### Sample Planet Images 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag.png)   

 ![Alt text]( https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag1.png)   

### 6) Create nightlight bins and label the daytime satellite imagery

The Gaussian Mixture Model is used to establish nightlight bins/labels to cluster the daytime satellite imagery into three categories based on nighttime values. The GMM-predicted cutoff values of **0.020** and **0.376** delineate a low nightlight bin, a medium nightlight bin, and a high nightlight bin. 

For a detailed implementation of the undersampling, go through [Nightlights_bins.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Nightlights_bins.ipynb) 

#### Steps:

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/classify_nightlight_bin2.png)  


#### Labeled Dataframre  

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/labled_table.png) 

#### Labeled Cluster Map

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/classify_nightlight_bin3.png)  

#### Labeled Satellite Images

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/label_low_mid_hight.png)   
 

### 7) Preparation of training and validation datasets

The preparation of the training and validation dataset involves employing a **stratified train-validation split method**, ensuring that each cluster group has a random assignment of samples to the train-validation set. This approach mitigates potential sampling issues, preventing situations where certain clusters lack training-validation data and ensuring a consistent sampling distribution. Specifically, an **80-20** split is implemented, with 
  - 80% of the data allocated for training
  - 20% for validation

For a detailed preparation of training and validation datasets, go through [Prepare_training_validation_datasets.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Prepare_training_validation_datasets.ipynb) 

#### Steps:

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/data_split.png)  

## Training and Validation Sets

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train_valid.png)  

### 8) Train variants of Convolutional Neural Network (CNN) models using a transfer learning approach

This project uses a novel **deep-learning** approach through a **transfer learning** method to predict consumption. Transfer learning is a technique that involves using a **pre-trained model** as a starting point for a new task. The pre-trained model has already learned to recognize many different features and can be used as a starting point for training a new model on a related task. It involves leveraging knowledge gained from one task to be repurposed for a different but related task. In this particular case, we are using nighttime light as a proxy for socioeconomic indicators. 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/tr.png)

Using pre-trained models (fine-tuning):

- It reduces computation costs
- It reduces computation time (high accuracy with few iterations)
- It requires less amount dataset(less data)
- It reduces carbon footprint (low environmental costs)
- It avoids training ML models from scratch

Our objectives are:
 
 1) Predict and assign the **nightlight bin labels (probability class)** of the satellite images (image classification using transfer learning).
 2) Simultaneously learn and **extract features** that are useful for consumption prediction (feature vector extraction).
    
The training process involves a series of steps using a **Deep Convolutional Neural Network** namely variants of [Virtual Geometry Group (VGG)](https://pytorch.org/vision/main/models/vgg.html) models such as **VGG-11**, **VGG-16**, and **VGG-19** as a Transfer Learning framework.

#### General Architecture of VGG Model 
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/vgg1.png)   

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/vgg.png)   

These models are renowned for their capabilities in **feature extraction** and **classification**. These models have been widely used in various applications, including **image classification**, **object recognition**, and **image segmentation**. These models were initially trained on the [ImageNet](https://image-net.org/update-mar-11-2021.php) dataset that contains over **1.2 million** images distributed across **1,000** classes. In the default configuration, the final network can classify images into 1000 object categories. The VGG architecture was introduced by Simonyan and Zisserman in 2014 from Oxford University, **Very Deep Convolutional Networks for Large Scale Image Recognition.** This model achieved a 92.7% top-5 test accuracy using the ImageNet dataset.


| Model      | Convolutional Layers | Fully Connected Layers | Parameters (approx.) | Pooling Layers  | Input Size | Activation Function | Pre-training   | Kernel Size | Stride | Padding |
|------------|----------------------|------------------------|----------------------|-----------------|------------|---------------------|----------------|------------|---------|---------|
| VGG-11     | 8                    | 3                      | 132 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |
| VGG-16     | 13                   | 3                      | 138 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |
| VGG-19     | 16                   | 3                      | 144 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |

#### Steps:
 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Train_image1.png)   

1) **Locate Image Data Directory**: Identify the directory containing image data (`data_dir`) previously created for the task.
  
2) **Download Pre-trained Models (VGG Models)**: Obtain pre-trained models, particularly VGG models, needed for the task. Link: https://pytorch.org/vision/main/models/vgg.html
  
3) **Set and Initialize Pre-trained Model**: Configure and initialize the pre-trained model with its initial parameters.

| Parameter               | Value                         |
|-------------------------|-------------------------------|
| Model names             | vgg11_bn, vgg16_bn, vgg19_bn  |
| Image classes           | 3  (low, medium, high)        |
| Batch size              | 8                             |
| Number of epochs        | 30                            |
| Feature extracting flag | True                          |               
| Input size              | 224x224x3     (H X W X C)     |

4) **Apply Image Transformation/Augmentation**:  To enhance the variety of training data (image).
     
| Transformation Step                           | Description                                                                                                       |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Image Flipping                            | Flips the image horizontally.                                                                                        |
| Image Cropping/Resizing                   | Resizes/crops the input image to the specified size.                                                                 |
| Image Normalization                       | Normalizes the input image by subtracting the mean value and dividing by the standard deviation [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]. |
| Conversion to PyTorch Tensors             | Converts images from PIL (Python Imaging Library) format to a PyTorch tensor format.                                |
| Rearrange Image Dimensions                | Dimensions change from HxWxC to CxHxW (channels first).                                                              |


5) **Create PyTorch Image Dataset**: Constructing PyTorch datasets serves to streamline image loading, facilitate efficient memory usage, and seamlessly integrate the image transformation module. Additionally, it automatically assigns labels to images based on the subdirectory structure.

6) **Create PyTorch Dataloader**: Set up a PyTorch data loader to divide the image dataset into batches for efficient training (Batching), randomize the order of data to prevent model bias (Shuffling), and accelerate data loading by fetching batches in parallel (Parallelizing Data Loading).

| Parameter                          | Value         |
|------------------------------------|---------------|
| Batch Size                         | 3             |
| Shuffling of the Data              | True          |
| Number of Workers for Data Loading | 4             |  

7) **Check CPU and GPU Availability**: Verify the availability of both CPU and GPU resources. Send the model to the appropriate device based on availability. For this project, high-performance GPUs, namely A100, V100, and T4, were used, each equipped with 51.0GB of RAM.
  
8) **Define Optimizer Function**: In this project a **Stochastic Gradient Descent (SGD)**, is used to update model parameters. The purpose of the optimizer function is to iteratively update the model's parameters in a way that minimizes the loss function. 

| Hyperparameter       | Value    |
|----------------------|----------|
| Momentum             | 0.1      |
| Learning Rate        | 1e-4     |
  
9) **Define Loss Function**: In this project **Categorical Cross Entropy (CCE)** loss function is used to quantify the difference between predicted and actual values. It Calculates the average cross-entropy loss between the predicted and true class labels. It is commonly used for multi-class classification problems.

10) **Train the model**: The model training involves executing all previously specified configurations, including initial parameters, image transformation, image datasets, data loaders, designated GPUs, optimizer function, and loss function.

11) **Save the Model**: The trained models are stored for future uses or deployment.
 
#### Models: The trained models can be found in this Google Drive directory:

- [VGG11 Trained Model](https://drive.google.com/file/d/10LwaTNbOrOtUzcAj1g8TgE6WgWdCXjQn/view?usp=sharing)
- [VGG16 Trained Model](https://drive.google.com/file/d/1rS05YsSy7U2D_lzbWQYzkR1uIzrGsesk/view?usp=sharing)
- [VGG19 Trained Model](https://drive.google.com/file/d/1mud2po2YebVlZh-2jrG6q4AFC94m0Qr3/view?usp=sharing)
 
12) **Evaluate the Model**: The table below displays the loss during both training and validation, including the accuracy of the model's performance.

| Model    | Train Loss | Valid Loss | Accuracy |
|----------|------------|------------|----------|
| VGG11     | 0.6014     | 0.5321    | 0.7718   |
| VGG16     | 0.5744     | 0.5284    | 0.7705   |
| VGG19     | 0.5805     | 0.5175    | 0.7849   |

The above results have indicated a good performance of the trained model in terms of correctly identifying each satellite image to its respective nightlight bin category/classes. Overall, the VGG models have been successfully trained using transfer learning to classify the satellite images to their respective nightlight bin classes. 

### Learning Curves
Learning curves represent the graphical depiction of a model's learning performance as a function of time. Widely employed as a diagnostic tool in machine learning, these curves are particularly useful for algorithms that progressively learn from a training dataset. 

Reviewing the learning curves of models during training can be used to diagnose problems with learning, such as an underfit or overfit model, as well as whether the training and validation datasets are suitably representative.

### Plot for Training and Validation Loss

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train-valid_loss.png)  

### Plot for Training and Validation Accuracy

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train_valid_acc.png)  

### 9) Feature Extraction and Aggregation

The feature vectors provide a lot of information about evidence of economic activity or lack of economic activity from satellite images. Feature vectors are a numerical representation of an object in an image. These features detected by the model include objects, edges, textures, and other patterns. In particular, urban areas, nonurban areas, roads, water bodies, agricultural areas, etc. 

For feature vector extraction each image passes through the pre-trained VGG model and the final dense layer is used to extract the feature vector from each image in the clustur with the output feature vector size of 4096.  Finally, the feature vectors of all images in the cluster are averaged to obtain a single feature vector per cluster. The cluster feature vector and cluster order files can be found [here](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/tree/main/Model_Output) 

For a detailed implementation of feature extraction, go through [Feature_extraction_aggregation.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Feature_extraction_aggregation.ipynb)

#### Steps:

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/feature_ext.png)  

## Visualizing Feature Maps (Activation Maps)  

Visualizing feature maps, also known as activation maps, provides valuable insights into how neural networks interpret and understand input data. 
Feature maps are representations of learned patterns and structures within the input data at different levels of abstraction.

### Sample feature maps from low nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_low.png)  

### Sample feature maps from medium nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_medium.png)  

### Sample feature maps from high nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_high.png)  

### 10) Building prediction model using Ridge regression model

In this project, a Ridge Regression model is employed to forecast consumption levels utilizing the feature vector extracted from the previous step. Ridge regression is a form of a linear regression model with L2 regularization that prevents overfitting. A Ridge Regression model is a supervised learning algorithm, that predicts a target variable based on one or more predictor features.

The feature vector, computed for each cluster, serves as the input variable, while the consumption level for each cluster is used as the output variable. Standardization or scaling is initially applied to both the input and output variables. Subsequently, a randomized cross-validation technique is employed with a 10-fold cross-validation, to predict consumption levels and evaluate model performance using a weighted R-square. 

#### Steps:

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/ridged_regression.png)

The ridge regression model has a cross-validated mean R-squared value of 0.5. The R-squared value indicates that the model explains 50% of the variance in the data. The mean absolute error (MAE) of 0.72 suggests that, on average, the model's predictions are off by 0.72 units from the actual values. The root mean squared error (RMSE) of 3.1 indicates that the model's predictions are off by 3.1 units on average.

#### Accuracy Metrics  

| Metric                        | Predicted consumption |
|-------------------------------|---------|
| Cross-validated mean R-squared | 0.5    |
| Mean Absolute Error (MAE)     | 0.72    |
| Root Mean Squared Error (RMSE)| 3.1     |


#### Cross-validated mean R-squared 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/cv_r2.png)   

For a detailed implementation of the Ridge regression model, go through [Predict_consumption_ridge_regression_model.ipynb](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Scripts/Predict_consumption_ridge_regression_model.ipynb)

Finally, the forecasted consumption levels were visually represented on a map.

#### Predicted consumption (without transformation)

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/predicted_con_direct.png)   
 
#### Predicted consumption (with log transformation)

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/predicted_con_log.png)   


#### Map of actual and estimated per capita consumption expenditure 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/CONS_IDW_ACT_PRE_PRELOG.png)   


## Conclusions/ Wayforward  

**Scalability and Versatility**: The proposed methodology extends its applicability beyond the prediction of consumption expenditure, it can be used to predict a wide range of socioeconomic indicators, including **poverty levels**, **wealth distribution**, **population density**, and **access to electricity**.

**Ease of Adoption and Adaptability**: 

- The trained **CNN/VGG models** and **extracted features** can be used by organizations, NSOs, and researchers to make predictions without the necessity to train the model from scratch and avoid the need for downloading and processing satellite imagery.

- Models developed for specific areas can be seamlessly transferred and applied to new regions with similar characteristics. Several previous studies indicated acceptable model performance in prediction.   

**Enhanced Reporting Capability**: This approach facilitates the generation of comprehensive **quarterly** and **annual reports**, addressing both temporal and spatial **data gaps**. It complements existing official statistical methods, contributing to more detailed socioeconomic reports.

**Decision Support Tools**: Country-specific dashboards can be built to visualize the spatial and temporal patterns of the model predictions. This feature supports evidence-based planning and interventions, providing valuable decision support for countries and NSOs.

**Exploration of Advanced Algorithms**: Testing advanced deep learning algorithms, including **ResNet**, **RegNet**, **EfficientNet**, **Inception V3**, and **AlexNet**,  etc holds the potential to further improve **prediction accuracy** and **model performance.**

**Hyperparameter Optimization**: Fine-tuning the model through optimization procedures, such as identifying optimal batch sizes and epochs, can enhance the model's overall performance, ensuring optimal model configuration.

**Integration of High-Resolution Images**: In this project, the [Planet's](https://www.planet.com/nonprofit/) high-resolution satellite images are used, incorporating other free open-source satellite images may improve or provide new insight into socioeconomic landscapes.

## Scripts 📜

The Script folder contains the following files: 

1) Survey_Data_Preprocessing_Malawi_2016.ipynb
2) GEE_Nightlight_2016_2017.js
3) Processing_Nighttime_Satellite_Imagery.ipynb
4) Generate_Image_Download_Locations.ipynb
5) Nightlights_bins.ipynb
6) Download_satellite_images_Planet.ipynb
7) Prepare_training_validation_datasets.ipynb
8) Image_labeling.ipynb
9) Train_VGG11_model_145.ipynb
10) Feature_extraction_aggregation.ipynb
11) Predict_consumption_ridge_regression_model.ipynb
12) Visualize_feature_maps.ipynb

## Processed/Intermediate output 🗂️

The data directory contains the processed output generated by the scripts:

1) df_clusters_malawi_2016.csv
2) df_clusters_malawi_nl.csv
3) df_malawi_download_loc.csv
4) df_malawi_loc_labed.csv
5) df_malawi_loc_labed_2016.csv
6) image_download_actual_malawi2016.csv
7) Extracted_feature_index_malawi_2016_VGG145.csv
8) predicted_malawi_2016_VGG145.csv


## Folder Structure of the project
```
 Main Project Folder/   
├── Images                           # Images used in the project and some output images   
├── Model_Output                     # Model output files such as feature vectors
├── Nighttime_Satellite_Imagery      # It contains the Nighttime Satellite Imagery used in this project
├── Processed_Files                  # Intermediate files from the analysis  
├── Scripts                          # Scripts used in the project mainly Jupyter notebooks and javascript codes
├── Survey Data                      # LSMS Survey Data: Fourth Integrated Household Survey data (Consumption Aggregate & Household Geovariables file)
├── README.md                        # Overview of the project  
├── License.txt                      # License for the project
```

## Issues

On the [issues](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/issues) page, the primary bugs in the code are documented along with their solutions. Additionally, you can create new issues if you encounter any new bugs.

## Packages required

**Python Packages**

| Library      | Description                                |
|--------------|--------------------------------------------|
| sys          | System-specific parameters and functions   |
| os           | Operating system interfaces                |
| Numpy        | Numerical operations and arrays            |
| Pandas       | Data manipulation and analysis             |
| Matplotlib   | Plotting library                           |
| Seaborn      | Statistical data visualization             |
| Plotly       | Interactive plots and dashboards           |
| Math         | Mathematical functions                     |
| Random       | Random number generation                   |
| Geoio        | Geospatial data I/O and processing         |
| IPython      | Interactive computing in Python            |
| Rasterio     | Geospatial raster data I/O and processing  |
| Utils        | Utility functions and tools                |
| torch        | PyTorch deep learning library              |
| torchvision  | PyTorch computer vision library            |
| time         | Time-related functions                     |
| copy         | Shallow and deep copy operations           |

## Contact

1) **Yonas Mersha**, Data Science Consultant | African Centre for Statistics (ACS) |  United Nations Economic Commission for Africa (UNECA)
 
2) **Issoufou Seidou Sanda**, Principal Statistician | African Centre for Statistics (ACS) |  United Nations Economic Commission for Africa (UNECA)

3) **ANJANA DUBE**, Senior Regional Advisor | African Centre for Statistics (ACS) |  United Nations Economic Commission for Africa (UNECA)
 
## Acknowledgments 

1) This research project  has been supported by **RCTP project**.
  
2) Special thanks to Jatin Mathur for sharing the create_space, add_nightlights, generate_download_locations, drop_zeros functions, and other code snippets on GitHub. We have integrated certain code snippets, tailoring them to suit the specific requirements of our project.

## References

### World Bank's Living Standards Measurement Study (LSMS) survey data 

National Statistical Office. Malawi - Fourth Integrated Household Survey 2016-2017, Ref. MWI_2016_IHS-IV_v02_M. Dataset downloaded from World Bank Microdata Library: https://microdata.worldbank.org/index.php/catalog/2936/related-materials.  


### VIIRS nighttime satellite imagery 

Elvidge, C.D, Zhizhin, M., Ghosh T., Hsu FC, Taneja J. Annual time series of global VIIRS nighttime lights derived from monthly averages:2012 to 2019. Remote Sensing 2021, 13(5), p.922, doi:10.3390/rs13050922 doi:10.3390/rs13050922

C.D. Elvidge, K. Baugh, M. Zhizhin, F. C. Hsu, and T. Ghosh, “VIIRS night-time lights,” International Journal of Remote Sensing, vol. 38, pp. 5860–5879, 2017.     

### Planet Satellite Imagery 

Planet Labs PBC. (2023). Planet Application Program Interface: In Space for Life on Earth. Retrieved from https://api.planet.com

## Related Works 

Neal Jean, Marshall Burke, Michael Xie, W. Matt Davis, David Lobell, and Stefano Ermon. 2016. "Combining satellite imagery and machine learning to predict poverty." Science 353, 6301. Combining satellite imagery and machine learning to predict poverty. GitHub: https://github.com/nealjean/predicting-poverty  


[Mapping-Poverty-With-Satellite-Images](https://github.com/huydang90/Mapping-Poverty-With-Satellite-Images)

[Jatin Mathur, Predicting Poverty Replication](https://github.com/jmather625/predicting-poverty-replication)

[COMBINING SATELLITE IMAGERY AND MACHINE LEARNING TO PREDICT POVERTY](https://sustain.stanford.edu/predicting-poverty)

[Satellite images can map poverty](https://www.science.org/content/article/satellite-images-can-map-poverty)
