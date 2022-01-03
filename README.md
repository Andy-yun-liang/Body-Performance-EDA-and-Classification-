# Body Performance Visualization and Classification Analysis


![image1](https://user-images.githubusercontent.com/73871814/147862750-58d7e4e0-d9a1-43dd-8c87-8f82705b6719.jpg)



## Table of Contents
   
* [Background](#background)
* [Visualizations](#visualizations)
   * [Summary Statistics](#summary-statistics)
   * [Observations](#observations)
* [Model-Preprocessing](#model-preprocessing)
* [Model Building](#model-building)
   * [XGBoost Model](#xgboost-model)
   * [SVM RBF Model](#svm-rbf-model)
* [Model Results](#model-results)


## Background

The objective of this project is to become more comfortable with multi-label classification with the tidymodels framework in R(Rstudio) and data visualization with the ggplot2 package and Tableau.

The body performance dataset is gathered by Seoul Olympic Commemorative Sports Promotion Agency and maintained by the Nationl Sports Promotion Agency, it can be found on https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=ace0aea7-5eee-48b9-b616-637365d665c1. 

In this classification analysis, we will be using a preprocessed version of the dataset found on https://www.kaggle.com/kukuroo3/body-performance-data.


## Visualizations 

Count of Female vs Male in the database. There are much more males in the database with a ratio of 2 male: 1 female.
![image](https://user-images.githubusercontent.com/95319198/147903318-209a0be7-7e96-4567-b1cf-754d38afc4e2.png)

Gif that visualizes the type of blood pressure that the individuals in the dataset have. 
![pogger_bp_gif](https://user-images.githubusercontent.com/95319198/147901343-6afbc86f-6b48-47ef-9980-87713a4918d0.gif)

Scatterplot of blood pressure of male and female within the four classes: A, B, C, and D. It is observed that men have a higher blood pressure than women.
![image](https://user-images.githubusercontent.com/95319198/147901417-442ea446-0790-41be-94b7-7ec1fb62524a.png)

Hexplot of blood pressure of the individuals in the database. It is observed that individuals with a lower blood pressure levels have a lower BMI.
![image](https://user-images.githubusercontent.com/95319198/147903680-e3b198e9-447e-463f-bd47-a6e8c9950ebd.png)

Scatterplot of BMI versus excercises between men and women within the four classes. Men generally perform better in all the excercises except for seated forward bend. Furthermore, individuals in class A outperform others of the same gender that are in the other classes. 
https://public.tableau.com/authoring/Book5_16401395973610/Sheet2#1
![image](https://user-images.githubusercontent.com/95319198/147901459-a1ff1064-615d-43dd-b144-003efee2ab71.png)

Hexplot of weight vs height by age. It is observed that most overweight individuals are of young age (<30).
![image](https://user-images.githubusercontent.com/95319198/147903280-0658e262-dd75-4a98-9b36-b18770b45f6f.png)

Looking at these boxplots of the classes with each of the response variables, we can see that there is a lot of outliers. 
![image](https://user-images.githubusercontent.com/95319198/147904406-8b429693-df47-49eb-9943-a183ae006a59.png)

Looking at distribution plots, we can see that a gaussian distribution. 
![image](https://user-images.githubusercontent.com/95319198/147906919-6b88dcc8-4be1-4b38-8183-cbcc83a21d14.png)

## Model Preprocessing
Now that are dataset is ready for model building, we split the dataset into train and test set. 

The testing set will be used for model evaluation after the tuning process.
```r
set.seed(123)
split = initial_split(df,prop = 0.7)
train_data = training(split)
test_data = testing(split)
```

In this step, we will be using tidymodel's recipe to remove low variance features and highly correlated features. Then scaled the predictor variables to mean 0 and standard deviation of 1. This processed is applied to the train set, and the test set transformation that follow these procedure will be automatically applied when we use the predict function.

1.Low variance features are removed because we don't want features that are constant and do not have any impact on the response variable

2.Highly correlated features are removed because we don't want multicollinearity in the models even though some models such as the random forest is relatively resistant. Multicollinearity can make our model very variable because it increases the variance of feature's coefficients.

3.Scaling is done so that distance based methods don't produce biased results

```r
dat_recipe = recipe(class~.,data=train_data) %>% 
                     step_nzv(all_predictors()) %>% 
                        step_corr(all_numeric(),-all_outcomes()) %>% 
                           step_normalize(all_predictors()) %>% 
                           prep()
```

## Model Building

Now that the data is preprocessed, we set up the folds.

```r
set.seed(123)
my_folds = bake(dat_recipe,new_data = train_data) %>% vfold_cv(v=5)

```

With the defined folds, we can now start building some classifiers. In this section, I will give a run down of how these models are setup. 

### XGBoost Example
First, define the model and choose the paramaters to tune. Tunable paramaters depend on the engine used.
```r
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

```r
XGB_wf = workflow() %>% add_recipe(dat_recipe) %>% add_model(XGBoost_mod)
XGB_tune_res = XGB_wf %>% tune_grid(resamples = my_folds,
                                  control = control_grid(save_pred=TRUE,save_workflow = TRUE),
                                  grid = XGB_grid
                                  )
```

To get the predictions, we need to finalize a model based on the "best" parameters that fits our needs then fit the split we initally created in the train-test split step
```r
#choosing the best parameters based on the roc_auc metric
best_XGB_params = XGB_tune_res %>% select_best(metric= "roc_auc")

#finalizing the model
final_XGB_mod = XGB_wf %>% finalize_workflow(best_XGB_params)

#fitting the model to the testing set
final_XGB_fit  = final_XGB_mod %>% last_fit(split)
```

With our final model, we can now collect the results and plot the roc_auc curve and confusion matrix
```r
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


### SVM RBF Example


#SVM rbf model

Initializing the model and the workflow
```r
svmRBF_mod =svm_rbf(mode = "classification",
                    engine="kernlab",
                    cost = tune(),
                    rbf_sigma = tune())

svmRBF_wf = workflow() %>% add_recipe(dat_recipe) %>% add_model(svmRBF_mod)
```

Setting up a default tune grid to get a baseline
```r
set.seed(123)
svmRBF_tune_res = svmRBF_wf %>% tune_grid(resamples = my_folds,
                                  control = control_grid(save_pred=TRUE),
                                  grid = 20
                                  )
```

Plotting the tuned parameters to see if the better parameters are on the edge or center.

```r
svmRBF_tune_res %>% collect_metrics() %>% filter(.metric == "roc_auc") %>%
  pivot_longer(cost:rbf_sigma, values_to = "val",names_to = "params") %>%
  ggplot(aes(val,mean,color = params)) + geom_point() + facet_wrap(~params,scales="free_x")

#Observation: we can see that lower cost values are preferred but can't really comment on the rbf_sigma parameter because there is a gap in data points
```

![svm_tune](https://user-images.githubusercontent.com/73871814/147897707-c3672bec-4cf0-44c2-a44a-fb544b8ecdc1.PNG)


Adjusting the tuning grid based on our observation
```r
svmRBF_grid = expand.grid(cost = c(0.5,1,2,3,4),rbf_sigma = c(0,0.1,0.2,0.4,0.5,0.7))

svmRBF_tune_res = svmRBF_wf %>% tune_grid(resamples = my_folds,
                                  control = control_grid(save_pred=TRUE,save_workflow = TRUE),
                                  grid = svmRBF_grid 
                                  )
```

Collect the best parameters and finalize the model and fit it to the test set
```r
best_svmRBF_params = svmRBF_tune_res %>% show_best(metric= "roc_auc")

final_svmRBF_mod = svmRBF_wf %>% finalize_workflow(best_svmRBF_params[1,])

final_svmRBF_fit  = final_svmRBF_mod %>% last_fit(split)
```

Collect the metrics from our finalize model and plot the roc_auc curve and confusion matrix
```r
#roc_auc and accuracy
final_svmRBF_fit %>% collect_metrics()

#auc plot of the test set
final_svmRBF_fit %>% collect_predictions(parameters = best_svmRBF_params) %>% roc_curve(class,.pred_A:.pred_D) %>% autoplot()

#get confusion matrix
(svmRBFtest_preds = final_svmRBF_fit %>% collect_predictions() %>% conf_mat(truth = class, estimate = .pred_class))

```
![svm_metrics](https://user-images.githubusercontent.com/73871814/147897937-e0513537-0edb-4f6d-86bb-2a0356e6b099.PNG)

![svm_rocauc_plot](https://user-images.githubusercontent.com/73871814/147897942-ce87f24f-66d4-4b32-b322-51eaa2b24ff7.PNG)

![svm_conf_mat](https://user-images.githubusercontent.com/73871814/147897945-5e4c83d6-b3fc-41c3-b9aa-6abef9708b59.PNG)


To see the rest of the tuning process for the other models used, check the appendix!

## Model Results

### Model performance summary

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
