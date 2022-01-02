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

In this step, we removed low variance features and highly correlated features. Then scaled the predictor variables to mean 0 and standard deviation of 1.

Low variance features are removed because we don't want features that are constant and do not have any impact on the response variable

Highly correlated features are removed because we don't want multicollinearity in the models even though some models such as the random forest is relatively resistant. Multicollinearity can make our model very variable because it increases the variance of feature's coefficients.

Scaling is done so that distance based methods don't produce biased results

```
dat_recipe = recipe(class~.,data=train_data) %>% 
                     step_nzv(all_predictors()) %>% 
                        step_corr(all_numeric(),-all_outcomes()) %>% 
                           step_normalize(all_predictors()) %>% 
                           prep()
```
## Model Building

Now that the data is preprocessed, we set up the folds.

```
set.seed(123)
my_folds = bake(dat_recipe,new_data = train_data) %>% vfold_cv(v=5)

```

With the defined folds, we can now start building some classifiers. In this section, I will give a run down of how these models are setup. 

First, define the model and choose the paramaters to tune. Tunable paramaters depend on the engine used.
```
XGBoost_mod = boost_tree( mode = "classification",
                      engine = "xgboost",
                      mtry = tune(),
                      trees = 800,
                      tree_depth = tune(),
                      min_n = tune(),
                      loss_reduction = tune(),
                      sample_size = tune(),
                      learn_rate = tune()
                      )
XGB_grid = grid_max_entropy(finalize(mtry(),train_data),
                            learn_rate(),
                            sample_size = sample_prop(),
                            min_n(),
                            tree_depth(),
                            loss_reduction(),
                            size=25)
```

Next, we define the work flow and use the tune_grid function to tune the parameters

```
XGB_wf = workflow() %>% add_recipe(dat_recipe) %>% add_model(XGBoost_mod)
XGB_tune_res = XGB_wf %>% tune_grid(resamples = my_folds,
                                  control = control_grid(save_pred=TRUE,save_workflow = TRUE),
                                  grid = XGB_grid
                                  )
```


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

