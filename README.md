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
   - [Classification Results](#result)
   



## Background

The objective of this project is to become more comfortable with multi-label classification with the tidymodels framework in R(Rstudio) and data visualization with the ggplot2 package and Tableau.

The body performance dataset is gathered by Seoul Olympic Commemorative Sports Promotion Agency and maintained by the Nationl Sports Promotion Agency, it can be found on https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=ace0aea7-5eee-48b9-b616-637365d665c1. 

In this classification analysis, we will be using a preprocessed version of the dataset found on https://www.kaggle.com/kukuroo3/body-performance-data.


## Visualizations 



## Model Preprocessing
Now that are dataset is ready for model building, we split the dataset into train and test set. 
The testing set will be used for model evaluation after the tuning process.
```{r}
set.seed(123)
split = initial_split(df,prop = 0.7)
train_data = training(split)
test_data = testing(split)
```

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

### XGB Example
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

To get the predictions, we need to finalize a model based on the "best" parameters that fits our needs then fit the split we initally created in the train-test split step
```
#choosing the best parameters based on the roc_auc metric
best_XGB_params = XGB_tune_res %>% select_best(metric= "roc_auc")

#finalizing the model
final_XGB_mod = XGB_wf %>% finalize_workflow(best_XGB_params)

#fitting the model to the testing set
final_XGB_fit  = final_XGB_mod %>% last_fit(split)
```

With our final model, we can now collect the results and plot the roc_auc curve and confusion matrix
```
#collect our accuracy and roc_auc score
final_XGB_fit %>% collect_metrics()

#auc plot
final_XGB_fit %>% collect_predictions(parameters = best_XGB_params) %>% roc_curve(class,.pred_A:.pred_D) %>% mutate(model = "XGB") %>% autoplot()

#plot confusion matrix
(XGB_test_preds = final_XGB_fit %>% collect_predictions() %>% conf_mat(truth = class, estimate = .pred_class))
```
![xgb_metrics](https://user-images.githubusercontent.com/73871814/147873769-9dc91b99-9fc4-408e-8b0c-e6f844f9f5e8.PNG)

![xgb_roc_auc_plot](https://user-images.githubusercontent.com/73871814/147873777-6c46dcf6-eb72-487b-bc08-168a88ffac2d.PNG)

![xgb_confusion_mat](https://user-images.githubusercontent.com/73871814/147873785-a508759a-6620-4217-b578-89397cf828cd.PNG)

To see the rest of the tuning process for the other models used, check the appendix!

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

(CatBoost and Ensembles on progress..)
