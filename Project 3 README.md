# Table of Contents
1. [Introduction- Churn Predictor Application](#Introduction)
2. [Data, Technology and Coding Standards](#Paragraph1)
   1. [Data Sources](#SubParagraph1)
   2. [Technology Stack](#Subparagraph2) 
   3. [Coding and Release Standards](#Subparagraph3)
3. [Analysis and Data Cleanse](#Paragraph2)
4. [Pre Processing Data](#Paragraph3)
5. [Model Development](#Paragraph4)
   1. [Identification](#SubParagraph4)
   2. [Training](#SubParagraph5)
   3. [Prediction](#Subparagraph6) 
   4. [Evaluation with Visualisations](#Subparagraph7)
5. [Web application](#Paragraph5)
6. [References](#Paragraph6)

<div style="page-break-after: always;"></div>

## Introduction- Churn Predictor Application <a name="Introduction"></a>

ChatCo is a nascent telecommunications company based in the US. The organisation believes that the cost of attrracting new customers is far higher than the cost of retaining customers.

Therefore the organisation is looking for an application which incorporates modern machine learning alogrithms to predict customer churn.

This project has developed a web based application that faciltates the processing of new customer data into a pre determined machine learning model to derive the churn/ retain value for the customer. 

## Data, Technology and Coding Standards <a name="paragraph1"></a>
### Data Sources <a name="subparagraph1"></a>

### Technology Stack <a name="subparagraph2"></a>


### Coding and Release Standards <a name="subparagraph3"></a>


## Analysis and Data Cleanse <a name="paragraph2"></a>

ChatCo has provided an initial though limited data set (via Kaggle) of its 7043 customers in the state of California. The project team has analysed this data set and determined that this represents a "Binary Classification type" machine learning problem. 

Based on the above analysis, the project team has performed the following data cleansing activities:

1. Eliminated data values that do not contribute to the decision making or data values that are derived values rather than True values. Data values that have been dropped are Customer ID, Count, Country, County, State, Lon lat, Partner, Churn Score, CLTV, Churn Reason.
2. Duplicates, if any, have been dropped.
3. Blank records for total charges- 11 of these- have been dropped.
4. Dataframe columns have been reassigned appropriate and correct headers.
5. Validation has been performed at every stage to ensure that the entire data set has been correctly loaded into the dataframe.

## Pre Processing Data <a name="paragraph3"></a>

The cleansed data set has then been pre processed i.e. prepared for the application of the machine learning models. This has included the below:

1. Selection of the target feature- Churn value has been selected as the target feature. 
2. Application of the Label encoder function to convert categorical and text data into numerical values. 
3. Application of the K Fold cross validation function to determine train and test data to avoid overfitting.
4. Application of Standard Scaler function across the data set. This was required to normalise count values of certain fields that were disproportionate. 


## Model Development <a name="paragraph4"></a>
Since the problem of predicting the churn value of a customer is a Binary Classification type, machine learning models suitable to this have been trained and developed.

The pipeline function has been used to automate the training, validation and prediction reporting of the selected models.

Below is an overview Machine Learning Model Procedure. Insert ppt slide


### Models <a name="subparagraph4"></a>
The following models have been trained and tested:

1. Logistic Regression
Logistic regression is a predictive modelling algorithm that is used when the Y variable is binary categorical in this instance Churn value i.e. it can take only two values like 1 or 0. The goal is to determine a mathematical equation that can be used to predict the probability of event 1 i.e. affirmative customer churn. 

2. Support Vector Machine
SVM or Support Vector Machine is a linear model for classification- this algorithm creates a line or a hyperplane which separates the data into classes.

3. Decision Tree
Decision Tree uses a flowchart like a tree structure to show the predictions that result from a series of feature-based splits. It starts with a root node and ends with a decision made by leaves.

4. Random Forest
Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

5. XGBoost
XGBoost is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

6. CatBoost


7. Deep Learning using Keras


### Model Training <a name="subparagraph5"></a>

The K fold cross validation technique has been used to determine Train and Test data for all models.
Based on prediction validation, hyper parameters have been further fine tuned. This is an iterative process which continues until model train well enough reducing the cost while increasing the accuracy. 

### Model Predictions <a name="subparagraph6"></a>
Below is the visual representation of the predictions for the various models: Insert Excel s/sheet

![Summary Results All Models](https://github.com/chirathlv/Project1/blob/Renu/Images/Total%20Wine%20Sales%20per%20Income%20Bracket.png)


### Model Evaluation <a name="subparagraph7"></a>
Based on theabove, it has been determined that XGBoost Model provides the best prediction. Hence this model has been deployed to the Churn Prediction Application.

The models predictions have been constrained by the rather limited data set that ChatCo has provided.


## Web Application <a name="paragraph5"></a>
As requested by ChatCo, a web application has been developed with a simple UI. The user can upload the data in csv format and then request the predictions for the successfully loaded data.

Given below is the Architecture overview for the application and the Wireframe.

Steps:
1. Store data per format in your local folder.
2. Select "   " to upload the data.
3. Verify that data upload is correct.
4. Select "Predict" to obtain the Churn value for the data loaded.
5. Review column 1 which display Customer Churn. 



## References <a name="paragraph6"></a>

1. [(https://www.kaggle.com/)]
