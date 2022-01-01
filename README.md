# Body Performance Visualization and Classification Analysis


## Table of Contents
   
   - [Background](#background)
   - [Dataset](#dataset)
   - [Visualizations](#visualizations)
   - * [Summary-Statistics](#summary-statistics)
   - * [Observations](#observations)
   - [Feature-Engineer](#feature-engineer)
   - [Parameter-Optimization](#parameter-optimization)
   - [Classification Results](#results)
   



## Background

The objective of this project is to become more comfortable with multi-label classification with the tidymodels framework in R(Rstudio) and data visualization with the ggplot2 package and Tableau.

## Dataset
This dataset is found at https://www.kaggle.com/kukuroo3/body-performance-data 


## Classification Results


| Model | ROC_AUC | Accuracy |
| :---  | :---:    |  :---:  |
| LightGBM   | 0.922    |  0.757|
| XGB     | 0.921       |  0.751|
| Random Forest | 0.919 |0.749|
|MLP Neural Net| 0.916|0.738| 
|SVM RBF| 0.886| 0.704|
|Multinomial | 0.846| 0.613|
|KNN | 0.845| 0.603|
|LDA | 0.841| 0.595|



