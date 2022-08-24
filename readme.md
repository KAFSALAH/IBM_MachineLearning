## Welcome to my IBM—ML Repository :smile:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

This repository aims is to build highly interpretable and accurate machine learning models that balance variance, bias, and time complexity. The Scikit-Learn framework is being used to build machine learning models and Keras for deep learning :bulb:

## Courses

Moreover, the repository contains hands-on labs of 6 machine learning courses created by IBM, which cover in-depth and breadth numerous ML concepts.

#### 01 - [Exploratory Data Analysis](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/01%20-%20Exploratory%20Data%20Analysis)
Hands-on Labs: SQL, Hypothesis Testing, Features Transformation, Scaling, Skewness & Importance.

#### 02 - [Supervised Machine Learning [Regression]](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/02%20-%20Supervised%20Machine%20Learning%20%5BRegression%5D)
Hands-on Labs: Cross-Validation, Ridge, Lasso, ElasticNet, Pipelines.

#### 03 - [Supervised Machine Learning [Classification]](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/03%20-%20Supervised%20Machine%20Learning%20%5BClassification%5D)
Hands-on Labs: Logistic Regression, K-Nearest Neighbor, Support Vector Machine, Decision Tree, Random Forrest, Extra Trees, Ensemble, Bagging, Boosting, Stacking, Model-Agnostic, Resampling Techniques.

#### 04 - [Unsupervised Machine Learning](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/04%20-%20Unsupervised%20Machine%20Learning)
Hands-on Labs: Principle Component Analysis, Distance Metrics, Inertia & Distortion, K-means, hierarchical, DBSCAN, Mean Shift Clustering.
 
#### 05 - [Deep Learning and Reinforcement Learning](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/05%20-%20Deep%20Learning%20and%20Reinforcement%20Learning)
Hands-on Labs: Gradient Descent, Backpropagation, Artificial NN, Convolutional NN, Recurrent NN.

#### 06 - [IBM ML Capstone Project — Online Courses Recommender System](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/06%20-%20Recommender%20Systems)
Hands-on Labs: Bag of Words, User-Profile Recommendation, Similarity-Index Recommendation.

## Capstone Projects
You are welcome to explore my findings in the personal capstone projects that I created during my learning journey.

#### A - [Treatment Costs per Person - Exploratory & Predictive Analysis](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/01%20-%20Exploratory%20Data%20Analysis/F%20-%20Treatment%20Costs%20per%20Person%20-%20Exploratory%20%26%20Predictive%20Analysis.ipynb)

• Aim: predict the cost of medical treatments based on six features, namely, age, sex, BMI, children, smoking status, and region.

• Procedure: in-Depth EDA via pair, bar, box, violin, and regression plots to see the effect of smoking on charges. Hypothesis testing the relationship between treatment costs above 35K$ and smoking status

• Findings: The test indicates that a person with 35K$ charges is most likely a smoker with a p-value = 0.023 and a confidence level of 0.977.

#### B - [Forecasting Photovoltaic Generated Power - Regression Analysis](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/02%20-%20Supervised%20Machine%20Learning%20%5BRegression%5D/F%20-%20Forecasting%20Photovoltaic%20Generated%20Power.ipynb)

• Aim: create a regression model that predicts the power generated by PV panels to facilitate energy management in power plants. 

• Procedure: Deployment of a pipeline that encompasses polynomial transformation, standard scaling, and regressor models. A GridSearchCV & hyper-parameters tuning of the algorithms along with benchmarking of Regular, Lasso, Ridge, Elastic Net & Gradient Boosting Regressors.

• Findings: The winner is the Gradient Boosting Regressor model with an R2 score of ~ 0.79.

#### C - [Fault Classification in Photovoltaic Plants - Multiclassification Analysis](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/03%20-%20Supervised%20Machine%20Learning%20%5BClassification%5D/J%20-%20Fault%20Classification%20in%20Photovoltaic%20Plants.ipynb)

• Aim: Classify the faults that might occur in photovoltaic Panels, namely, Short-Circuit, Open-Circuit, Degradation, and Shadowing.

• Procedure: Stratified split & features scaling. Re-weighting the imbalanced classes. GridSearch & hyper-parameters tuning of Logistic Regression, Decision Tree & Random Forrest along with benchmarking the three algorithms.

• Findings:  The winner is Decision Tree algorithm with an accuracy and a weighted F1-score of ~ 97%.

#### D - [Date Fruit Segmentation & Dimensionality Reduction via PCA](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/04%20-%20Unsupervised%20Machine%20Learning/F%20-%20Date%20Fruit%20Segmentation%20%26%20Dimensionality%20Reduction%20via%20PCA.ipynb)

• Aim: Cluster dates based on their physical features. 

• Procedure: Multicollinearity check and Data scaling. Reducing the number of features via PCA. Comparative analysis between K-means, Agglomerative, Mean Shift & DBSCAN clustering.  

• Findings: The winner is k-means++ technique.

#### E -  [MRI Brain Tumor Classification via CNN](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/05%20-%20Deep%20Learning%20and%20Reinforcement%20Learning/J%20-%20MRI%20Brain%20Tumor%20Classification%20via%20CNN.ipynb)

• Aim: Detect whether a patient has a brain tumor.

• Procedure: Image Scaling. Train a CNN model to classify brain tumors. Deploy the deep learning model using flask app.

• Findings: The CNN model accuracy is 97%.

#### F - [Personalized Course Recommendation System for Data Science Learners.](https://github.com/KAFSALAH/IBM_MachineLearning/tree/main/06%20-%20Recommender%20Systems)

• Aim: To build a recommendation system that recommends the most suitable courses for learners on educational platforms.

• Procedure: Several techniques are used to build the recommendation system. Listed in findings as bellow, 

• Findings: The recommender system is created via several approaches as follows

Appraoch 1 - [Content-based Course Recommender System Using User Profile and Course Genres](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/06%20-%20Recommender%20Systems/D%20-%20Content-Based%20User-Profile.ipynb)

Approach 2 - [Content-based Course Recommender System using Course Similarities](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/06%20-%20Recommender%20Systems/E%20-%20Content-Based%20Similarity-Index.ipynb)

Approach 3 - [Content-Based PCA Clustering Course Recommender System](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/06%20-%20Recommender%20Systems/F%20-%20Content-Based%20PCA-Clustering.ipynb)

Approach 4 - [Collaborative Filtering based Recommender System using K Nearest Neighbor
](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/06%20-%20Recommender%20Systems/G%20-%20Collaborative-Filtering%20via%20KNN.ipynb)

Approach 5 - [Collaborative Filtering based Recommender System using Non-negative Matrix Factorization](https://github.com/KAFSALAH/IBM_MachineLearning/blob/main/06%20-%20Recommender%20Systems/H%20-%20Collaborative-Filtering%20via%20NMF.ipynb)
