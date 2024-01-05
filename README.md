# Investigating Machine Learning/Deep Learning Approaches for Predicting Socioeconomic Indicators through Transfer Learning: A Fusion of Satellite Imagery and Survey Data

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

The primary aim of this project is to integrate **survey data**,  **nighttime satellite imagery**, and  **daytime high-resolution satellite imagery**, leveraging **machine learning/deep Learning** methodologies to forecast **socioeconomic indicators** using  a reproducible framework utilizing open-source tools and resources (i.e. Consumption expenditure  for Malawi for the year 2016).​

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

We generated download locations for daytime satellite images by employing both systematic and stochastic sampling approaches within cluster bounding boxes. This approach enabled us to generate 50 images per cluster, corresponding to 50 grids within the bounding box, ensuring a diverse and representative sample of imagery across the cluster. For each set of 50 points, we constructed the image name, image latitude, and image longitude, appending them to the data frame. We undertook undersampling in areas with low/no nightlights to ensure class imbalance in the dataset.

### 4) Create nightlight bins/labels

The Gaussian Mixture Model is used to establish nightlight bins/labels to classify the daytime satellite imagery into three categories based on nighttime values. The GMM-predicted cutoff values of 0.020 and 0.376 delineate a low nightlight bin, a medium nightlight bin, and a high nightlight bin. The respective percentages of daytime satellite imagery within each bin are 16,800 (49.55%), 10,250 (30.23%), and 6,850 (20.20%). This leads to a cumulative image count of 33,900.

#### Sample Labeled Imagery

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/label_low_mid_hight.png)   
 
### 5) Daytime satellite imagery acquisition and processing

The acquisition of high-resolution daytime satellite imagery is performed using the Planet API, which provides images specifically for research and academic purposes. The Planet Scope (PSScene) images have a spatial resolution ranging from 3.7 to 4.1 meters, later resampled to 3 meters for practical use. The process of obtaining Planet Imagery encompasses a series of steps. Initially, we set up the API Key in Planet Explorer. Following this, we apply essential filters such as geometry, date, and cloud filters to download the images properly. The download locations (image latitude and longitude) derived from previous steps serve as inputs for image retrieval, incorporating additional parameters like a zoom level of 14 and a maximum cloud filter of 0.05 (5%). The image acquisition spans the period from 2016 to 2017, culminating in a total of 33,900 downloaded images.

#### Sample Planet Images 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag.png)   

 ![Alt text]( https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag1.png)   

### 6) Preparation of training and validation datasets

The preparation of the training and validation dataset involves employing a stratified train-validation split method, ensuring that each cluster group has a random assignment of samples to the train-validation set. This approach mitigates potential sampling issues, preventing situations where certain clusters lack training-validation data and ensuring a consistent sampling distribution. Specifically, an 80-20 split is implemented, with 80% of the data allocated for training and 20% for validation within each cluster.


## Training and validation datasets

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train_valid.png)  

### 7) Train variants of Convolutional Neural Network (CNN) models using a transfer learning approach

This project uses a novel deep-learning approach through a transfer learning method to predict consumption. Transfer learning is a technique that involves using a pre-trained model as a starting point for a new task. The pre-trained model has already learned to recognize many different features and can be used as a starting point for training a new model on a related task. It involves leveraging knowledge gained from one task to be repurposed for a different but related task. In this particular case, we are using nighttime light as a proxy for socioeconomic indicators. Our objective is to predict the probability class of a given daytime satellite image and assign it to the appropriate nightlight bin category, simultaneously learning features that are useful for consumption prediction. 

The training process involves a series of steps using variants of CNN pre-trained [Virtual Geometry Group (VGG) models](https://pytorch.org/vision/main/models/vgg.html) The project employs very deep CNNs for Large-Scale Image Recognition, such as VGG-11, VGG-16, and VGG-19. 


#### VGG Model Architecture
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/vgg_model.png)   


The models used in this project are very Deep Convolutional Networks for Large-Scale Image Recognition such as **VGG-11**, **VGG-16**, and **VGG-19**. These models are renowned for their capabilities in feature extraction and classification. The VGG models are known for their depth and their ability to capture intricate features from images, making them suitable for a wide range of image classification tasks. These models have been widely used in various applications, including image classification, object recognition, and image segmentation. These models were initially trained on the [ImageNet](https://image-net.org/update-mar-11-2021.php) dataset that contains over 1.2 million images distributed across 1,000 classes. 


| Model      | Convolutional Layers | Fully Connected Layers | Parameters (approx.) | Pooling Layers  | Input Size | Activation Function | Pre-training   | Kernel Size | Stride | Padding |
|------------|----------------------|------------------------|----------------------|-----------------|------------|---------------------|----------------|------------|---------|---------|
| VGG-11     | 8                    | 3                      | 132 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |
| VGG-16     | 13                   | 3                      | 138 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |
| VGG-19     | 16                   | 3                      | 144 million          | Max Pooling     | 224x224x3  | ReLU                | ImageNet       | 3 X 3      |   1     |   1     |

- The modeling process begins with model initialization, downloading pre-trained models, and configuring their initial parameters.
- We tailor the model architecture to our task, incorporating RGB Planet Scope satellite images (3 channels) with a width and height of 224 by 224 dimensions. 
- Data augmentation techniques are then applied to enhance dataset variability, including image flipping, resizing, cropping, normalization, and conversion to PyTorch tensors.
- Data loaders are established for efficient data handling during training and validation.
- The Stochastic Gradient Descent (SGD) optimizer function is employed with a momentum of 0.1 and a learning rate of 1e-4.
- The Categorical Cross Entropy (CCE) loss function is chosen for its suitability in multi-class classification problems.  
- The training process begins with the following batch size, epochs, and output classes.
  
| Parameter      | Value |
|----------------|-------|
| Batch Size     | 8     |
| Epochs         | 30    |
| Output Classes | 3     |

- Subsequently, the model's performance is evaluated using the following accuracy metrics: Train Loss, Valid Loss, and Accuracy.
## Models and their accuracy in classifying daytime satellite images into predefined nightlight categories or labels

### Training and Validation Loss

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train-valid_loss.png)  

### Training and Validation Accuracy

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/train_valid_acc.png)  

| Model    | Train Loss | Valid Loss | Accuracy |
|----------|------------|------------|----------|
| VGG11     | 0.6014     | 0.5321     | 0.7718   |
| VGG16     | 0.5744     | 0.5284     | 0.7705   |
| VGG19     | 0.5805     | 0.5175     | 0.7849   |

- Finally, the trained model is saved for future use.

#### Models
The trained models can be found in this Google Drive directory:

1) [VGG11 Trained Model](https://drive.google.com/file/d/10LwaTNbOrOtUzcAj1g8TgE6WgWdCXjQn/view?usp=sharing)

### 8) Feature extraction and aggregation

The feature vectors provide a lot of information about evidence of economic activity or lack of economic activity from satellite images. Feature vectors are a numerical representation of an object in an image. These features detected by the model include objects, edges, textures, and other patterns. In particular, urban areas, nonurban areas, roads, water bodies, etc. For feature vector extraction each image passes through the pre-trained VGG model and the final dense layer is used to extract the feature vector from each image in the clustur with the output feature vector Size of 4096.  Finally, the feature vectors of all images in the cluster are averaged to obtain a single feature vector per cluster. The cluster feature vector and cluster order files can be found [here](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/tree/main/Model_Output) This feature vector is then used as input to a Ridge Regression model, which is used to predict consumption levels for that cluster.

## Visualizing Feature Maps (Activation Maps)  

### Sample feature maps from low nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_low.png)  

### Sample feature maps from medium nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_medium.png)  

### Sample feature maps from high nightlight bin

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/fm_high.png)  

### 9) Building prediction model using Ridge regression model

In this project, a Ridge Regression model is employed to forecast consumption levels utilizing the feature vector extracted from the previous step. Ridge regression is a form of a linear regression model with L2 regularization that prevents overfitting. A Ridge Regression model is a supervised learning algorithm, that predicts a target variable based on one or more predictor features. The feature vector, computed for each cluster, serves as the input variable, while the consumption level for each cluster is used as the output variable. Standardization or scaling is initially applied to both the input and output variables. Subsequently, a randomized cross-validation technique is employed with a 10-fold cross-validation, to predict consumption levels and evaluate model performance using a weighted R-square. The Ridge Regression model yielded a cross-validated mean R-squared value of 0.5. Finally, the forecasted consumption levels were visually represented on a map as well as the feature maps/ activation maps.

#### Predicted consumption (without transformation)

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/predicted_con_direct.png)   
 
#### Predicted consumption (with log transformation)

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/predicted_con_log.png)   


#### Map of actual and estimated per capita consumption expenditure 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/CONS_IDW_ACT_PRE_PRELOG.png)   

#### Cross-validated mean R-squared 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/cv_r2.png)   

#### Accuracy Metrics  

| Metric                        | Predicted consumption |
|-------------------------------|---------|
| Cross-validated mean R-squared | 0.5    |
| Mean Absolute Error (MAE)     | 0.72    |
| Root Mean Squared Error (RMSE)| 3.1     |

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

## Processed output 🗂️

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

**Yonas Mersha**, Data Science Consultant | African Centre for Statistics (ACS) |  United Nations Economic Commission for Africa (UNECA)
 
**Issoufou Seidou Sanda**, Principal Statistician | African Centre for Statistics (ACS) |  United Nations Economic Commission for Africa (UNECA)
 
## Acknowledgments 

This research project  has been supported by **RCTP project**. 

Special thanks to Jatin Mathur for sharing the create_space, add_nightlights, generate_download_locations, drop_zeros functions, and other code snippets on GitHub. We have integrated certain code snippets, tailoring them to suit the specific requirements of our project.

## References

### World Bank's Living Standards Measurement Study (LSMS) survey data 

National Statistical Office. Malawi - Fourth Integrated Household Survey 2016-2017, Ref. MWI_2016_IHS-IV_v02_M. Dataset downloaded from World Bank Microdata Library: [https://microdata.worldbank.org/index.php/catalog/2936/related-materials].  


### VIIRS nighttime satellite imagery 

Elvidge, C.D, Zhizhin, M., Ghosh T., Hsu FC, Taneja J. Annual time series of global VIIRS nighttime lights derived from monthly averages:2012 to 2019. Remote Sensing 2021, 13(5), p.922, doi:10.3390/rs13050922 doi:10.3390/rs13050922

C.D. Elvidge, K. Baugh, M. Zhizhin, F. C. Hsu, and T. Ghosh, “VIIRS night-time lights,” International Journal of Remote Sensing, vol. 38, pp. 5860–5879, 2017.     

### Planet Satellite Imagery 

Planet Labs PBC. (2023). Planet Application Program Interface: In Space for Life on Earth. Retrieved from [https://api.planet.com]

## Related Works 

Jatin Mathur, Predicting Poverty Replication, GitHub: [https://github.com/jmather625/predicting-poverty-replication]  


Neal Jean, Marshall Burke, Michael Xie, W. Matt Davis, David Lobell, and Stefano Ermon. 2016. "Combining satellite imagery and machine learning to predict poverty." Science 353, 6301. Combining satellite imagery and machine learning to predict poverty. GitHub: [https://github.com/nealjean/predicting-poverty]

## Related projects 

Link: [https://sustain.stanford.edu/predicting-poverty]

Link: [https://www.science.org/content/article/satellite-images-can-map-poverty]
