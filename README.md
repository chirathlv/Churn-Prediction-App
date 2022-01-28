# App Demo

![App Demo](https://github.com/chirathlv/Churn-Prediction-App/blob/pre-prod/Images/Demo.gif)

# Table of Contents

1. [Introduction- Churn Predictor Application](#Introduction)
2. [Data, Technology and Coding Standards](#Paragraph1)
   1. [Data Sources](#SubParagraph1)
   2. [Technology Stack](#Subparagraph2)
   3. [Technical Solution](<(#Subparagraph3)>)
   4. [Coding and Release Standards](#Subparagraph4)
3. [Analysis and Data Cleanse](#Paragraph2)
4. [Data Pre Processing](#Paragraph3)
5. [Model Development](#Paragraph4)
   1. [Machine Learning Overview](#SubParagraph5)
   2. [Models](#SubParagraph6)
   3. [Model Training](#SubParagraph7)
   4. [Model Prediction](#Subparagraph8)
   5. [Model Evaluation](#Subparagraph9)
6. [Web Application](#Paragraph5)
7. [References](#Paragraph6)

<div style="page-break-after: always;"></div>

## Introduction- Churn Predictor Application <a name="Introduction"></a>

ChatCo is a nascent telecommunications company based in the US. The organisation believes that the cost of attracting new customers is far higher than the cost of retaining customers.

Therefore the organisation is looking for an application which incorporates modern machine learning algorithms to predict customer churn.

This project has developed a web based application that facilitates the processing of new customer data into a pre-determined machine learning model to derive the churn/ retain value for the customer.

## Data, Technology and Coding Standards <a name="paragraph1"></a>

### Data Sources <a name="subparagraph1"></a>

Telco customer churn: IBM dataset

This is IBM Cognos Analytics 11.1.3+ base samples dataset. Following is the summary of it.

| Feature             | Description                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `CustomerID`        | A unique ID that identifies each customer                                                                           |
| `Count`             | A value used in reporting/dashboarding to sum up the number of customers in a filtered set                          |
| `Country`           | The country of the customer's primary residence                                                                     |
| `State`             | The state of the customer's primary residence                                                                       |
| `City`              | The city of the customer's primary residence                                                                        |
| `Zip Code`          | The zip code of the customer's primary residence                                                                    |
| `Lat Long`          | The combined latitude and longitude of the customer's primary residence                                             |
| `Latitude`          | The latitude of the customer's primary residence                                                                    |
| `Longitude`         | The longitude of the customer's primary residence                                                                   |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Senior Citizen`    | Indicates if the customer is 65 or older: Yes, No                                                                   |
| `Partner`           | Indicate if the customer has a partner: Yes, No                                                                     |
| `Dependents`        | Indicates if the customer lives with any dependents: Yes, No                                                        |
| `Tenure Months`     | Indicates the total amount of months that the customer has been with the company by the end of the quarter          |
| `Phone Service`     | Indicates if the customer subscribes to home phone service with the company: Yes, No                                |
| `Multiple Lines`    | Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No                          |
| `Internet Service`  | Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable              |
| `Online Security`   | Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No      |
| `Online Backup`     | Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No        |
| `Device Protection` | Indicates if the customer subscribes to an additional device protection plan for their Internet equipment : Yes, No |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |
| `Gender`            | The customer's gender: Male, Female                                                                                 |

### Technology Stack <a name="subparagraph2"></a>

### Technical Solution <a name="subparagraph3"></a>

### Coding and Release Standards <a name="subparagraph4"></a>

## Analysis and Data Cleanse <a name="paragraph2"></a>

ChatCo has provided an initial though limited data set (via Kaggle) of its 7043 customers in the state of California. The project team has analysed this data set and determined that this represents a "Binary Classification type" machine learning problem.

Based on the above analysis, the project team has performed the following data cleansing activities:

1. Eliminated data values that do not contribute to the decision making or data values that are derived values rather than True values. Data values that have been dropped are Customer ID, Count, Country, County, State, Lon lat, Partner, Churn Score, CLTV, Churn Reason.
2. Duplicates, if any, have been dropped.
3. Blank records for total charges- 11 of these- have been dropped.
4. Dataframe columns have been reassigned appropriate and correct headers.
5. Validation has been performed at every stage to ensure that the entire data set has been correctly loaded into the dataframe.

## Data Pre Processing <a name="paragraph3"></a>

The cleansed data set has then been pre processed i.e. prepared for the application of the machine learning models. This has included the below:

1. Selection of the target feature- Churn value has been selected as the target feature.
2. Application of the Label Encoder function to convert categorical and text data into numerical values.
3. Application of the K Fold cross validation function to determine train and test data to avoid overfitting.
4. Application of Standard Scaler function across the data set. This was required to normalise count values of certain fields that were disproportionate.
5. Application of oversampling tools to correct the imbalanced data.

## Model Development <a name="paragraph4"></a>

Since the problem of predicting the churn value of a customer is a Binary Classification type, machine learning models suitable to this problem have been trained and developed.

The pipeline function has been used to automate the training, validation and prediction reporting of the selected models.

## Machine Learning Overview <a name="subparagraph5"></a>

Below is an overview Machine Learning Model Procedure.
![Machine Learning Overview](https://github.com/chirathlv/Churn-Prediction-App/blob/pre-prod/Images/Machine%20Learning%20Workflow.png)

### Models <a name="subparagraph6"></a>

The following seven models have been trained and tested:

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

### Model Training <a name="subparagraph7"></a>

The K fold cross validation technique has been used to determine Train and Test data for all models.
Based on the prediction validation, hyper parameters have been further fine tuned. This is an iterative process which continues until the model has been trained well enough, reducing the cost of running the model while increasing its accuracy.

### Model Predictions <a name="subparagraph8"></a>

Below is the visual representation of the predictions for the various models:

![Summary Results- All Models](https://github.com/chirathlv/Churn-Prediction-App/blob/pre-prod/Images/Model%20Results.PNG)

![Accuracy Score Summary- All Models](https://github.com/chirathlv/Churn-Prediction-App/blob/pre-prod/Images/Accuracy%20Score%20Graph.PNG)

### Model Evaluation <a name="subparagraph9"></a>

Based on the above, it has been determined that XGBoost Model provides the best prediction. Hence this model has been deployed to the Churn Prediction Application.

The models' predictions have been constrained by the rather limited data set that ChatCo has provided.

## Web Application <a name="paragraph5"></a>

As requested by ChatCo, a web application with cloud based RestAPI has been developed. The simple User Interface requires the user to upload the data in a predetermined csv format followed by a request to predict the churn value for the successfully loaded data.

Given below are the key user instructions for this web application:

Steps:

1. Store data per format in your local folder.
2. Select " " to upload the data.
3. Verify that data upload is correct.
4. Select "Predict" to obtain the Churn value for the data loaded.
5. Review column 1 which displays Customer Churn.
6. Apply filters to obtain a more granular view of the data.

## References <a name="paragraph6"></a>

1. [(https://www.kaggle.com/)]
