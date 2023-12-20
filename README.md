# ML/Deep Learning for Socioeconomic Indicator Prediction

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/intro_im.png)   

The primary aim of this project is to integrate **survey data** and **high-resolution satellite imagery**, leveraging **machine learning/Deep Learning** methodologies to forecast **socioeconomic indicators**. 
The project is specifically centered around the theme "**Application of Deep Learning to Predict Socioeconomic Indicators using Survey Data and High-resolution Satellite Imagery."** 

Case Study: **"Consumption Expenditure Prediction in Malawi."**

Using such an approach demands a substantial grasp of machine learning and Deep Learning concepts, coupled with a considerable investment of time in data preparation and model training. 

To streamline this complex process, we documented each step, explaining each line of code, adopting the contemporary methodology established by researchers at [Stanford University.]( https://sustain.stanford.edu/predicting-poverty)

Our overarching goal is to create a reproducible framework utilizing open-source tools and resources. This framework is designed to empower National Statistics Offices (NSOs) across Africa, providing them with the means to implement similar methodologies.

The documentation includes important stages such as:
1) **Survey data collection, and processing**
2) **Nightlight satellite imagery acquisition, and processing**
3) **Sampling method to generate download locations and undersampling to avoid bias**
4) **Create nightlight bins/labels**
5) **Daytime satellite imagery acquisition and processing**
6) **Preparation of training and validation datasets**
7) **Train Convolutional Neural Network (CNN) models and their variants**
8) **Feature extraction**
9) **Prediction model**

The implementation is facilitated through the 
- **Python programming language**
- **Google Earth Engine**
- **PyTorch framework**

This endeavor aims not only to advance our understanding of the subject matter but also to foster knowledge-sharing and collaboration within the data science community.

---

## Packages required

**Python Packages**:
- sys
- os
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Math
- Random
- Geoio
- IPython
- Rasterio
- Utils
- torch
- torchvision
- time
- copy

## Procedure 

## Scripts

The Script folder contains the following file: 

1) Survey_Data_Preprocessing_Malawi_2016.ipynb
2) GEE_Nightlight_2016_2017.js
3) Processing_Nighttime_Satellite_Imagery.ipynb
4) Generate_Image_Download_Locations.ipynb
5) Nightlights_bins.ipynb
6) Download_satellite_images_Planet.ipynb
7) Prepare_training_validation_datasets.ipynb
8) Train_VGG11_model_145.ipynb
9) Feature_extraction_aggregation.ipynb

## Processed output 

The data directory contains the processed output generated by the scripts:

1) df_clusters_malawi_2016.csv
2) df_clusters_malawi_nl.csv
3) df_malawi_download_loc.csv
4) df_malawi_loc_labed.csv
5) df_malawi_loc_labed_2016.csv
6) image_download_actual_malawi2016.csv

## VGG Model Architecture
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/vgg_model.png)   

## Model Output 
The [model output](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/tree/main/Model_Out) directory contains the aggregated extracted feature vectors and cluster order files:

  ### Aggregated extracted feature vectors: 
     - used as input to predict consumption at a cluster level.
  ### Cluster order: 
  - the order of the clusters during the feature extraction.

## Models

The trained models can be found in this Google Drive directory:

1) [VGG11 Trained Model](https://drive.google.com/file/d/10LwaTNbOrOtUzcAj1g8TgE6WgWdCXjQn/view?usp=sharing)

## Daily Consumption per Capita for Malawi for 2016

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Daily_Per_Capita_Consumption_Malawi_Year_2016.png)   

## Nightlight and Consumption for Malawi for 2016
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/nightlight_consumtion.png) 

## Cluster Boxes
![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Cluster_box.png)  

## Sample Nightlight Image 

![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/nightlight.png)  

## Sample Planet Images 

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag.png)   

 ![Alt text]( https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Planet_imag1.png)   

## Sample Labeled Imagery

 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Low_nighttime.png)   
 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/Medium_nighttime.png)   
 ![Alt text](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/blob/main/Images/High_nighttime.png)


## Issues

On the [issues](https://github.com/YonSci/UNECA-Deep-Learning-for-Socioeconomic-Indicator-Prediction/issues) page, the primary bugs in the code are documented along with their solutions. Additionally, you can create new issues if you encounter any new bugs.

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
