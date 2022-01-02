# Body Performance Visualization and Classification Analysis


![image1](https://user-images.githubusercontent.com/73871814/147862750-58d7e4e0-d9a1-43dd-8c87-8f82705b6719.jpg)



## Table of Contents
   
   - [Background](#background)
   - [Visualizations](#visualizations)
   - * [Summary-Statistics](#summary-statistics)
   - * [Observations](#observations)
   - [Feature-Engineer](#feature-engineer)
   - [Model-Preprocessing](#model-preprocessing)
   - [Model Building](#model-building)
   - [Classification Results](#results)
   



## Background

The objective of this project is to become more comfortable with multi-label classification with the tidymodels framework in R(Rstudio) and data visualization with the ggplot2 package and Tableau.

The body performance dataset is gathered by Seoul Olympic COmmemorative Sports Promotion Agency and maintained by the Nationl Sports Promotion Agency, it can be found on https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=ace0aea7-5eee-48b9-b616-637365d665c1. 

In this classification analysis, we will be using a preprocessed version of the dataset found on https://www.kaggle.com/kukuroo3/body-performance-data.


## Visualizations 



## Model Preprocessing

For model preprocessing, we removed low variance features and highly correlated features. Then scaled the predictors variables to mean 0 and standard deviation of 1.

Low variance features are removed because we don't want features that are constant and do not have any impact on the response variable

Highly correlated features are removed because 

Scaling is done so that distance based methods don't produce biased results

```
dat_recipe = recipe(class~.,data=train_data) %>% 
                     step_nzv(all_predictors()) %>% 
                        step_corr(all_numeric(),-all_outcomes()) %>% 
                           step_normalize(all_predictors()) %>% 
                           prep()
```
## Model Building







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

![roc_auc](https://user-images.githubusercontent.com/73871814/147862865-dc3babae-c1dc-4988-a288-d87125322b88.PNG)


