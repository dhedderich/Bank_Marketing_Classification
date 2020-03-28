# UCI Bank Marketing Dataset - Classification 

The project is based on the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and compares various tree based algorithms using scikit-learn and XGBoost with two different neural network architectures utilizing both the sequential and the functional API of Keras regarding the binary classification task. The neural network implemented via the sequential API of Keras is a fully connected neural network while the second neural network divides the inputs into two different category types. The newtork architecture is split into two separate fully connected layer architectures whereafter they are concatenated to feed multiple final fully connected layers. This architecture is implemented via the functional API of Keras.

## Dataset

The description of the features can be found on [UCI Bank Marketing Dataset Website](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The "bank-additional-full.csv" file was used as input file. 

## Structure of the Jupyter notebook

The Jupyter notebook can be divided into 11 steps that deal with the initial dataset analysis, feature engineering and model building/results.

# 1. First Impression of the dataset

# 2. Univariate analysis of the dataset

- Value count analysis of the categorical features
- Distribution analysis of the numerical features
- Value count analysis of the target feature (Low amount of positive feature values leads to F1 score as central metric)

# 3. Bivariate analysis of the dataset

- Pearson's correlation analysis of the numerical features regarding the target feature
- Barplot analysis of the numerical features regarding the target feature (seaborn)
- Barplot analysis of the categorical features regarding the taget feature (seaborn)
- Correlation analysis of the categorical feature using chi-squared test and mutual information (scikit-learn)

# 4. Detect abnormal and missing values

- Estimation of the amount of data cleaning via NaN value counts for all features - The decision rule is based on the low amount of datapoints (41.118 instances).
- Boxplot analysis of the numerical features regarding outliers (seaborn)
- Estimation of how many instances would be lost when using IQR with various multiplicators (1.0, 2.0, 2.5)

# 5. Feature Engineering

Implementation of the findings of the preceding analysis steps:
- Removal of the feature "Default"
- Removal of NaN values (7% of instances of the dataset)
- Adding the new feature "contacted" stating if customers where previously contacted or not, extending the information of the feature "pdays"
- Combining "emp.var.rate" and "nr.employed" into one feature "employment"

# 6. Data preparation pipelines - tree based algorithms

Numerical features are not preprocessed due to the focus on tree based algorithms. As in recent month there are several analysis that show that OneHotEncoding damages the results of tree based algorithms, it is not used for nominal categorical features. Instead, they are ordinal encoded without predecessing order using OrdinalEncoder. Ordinal categorical features are preprocessed using OrdinalEncoder with the respective implementation of their order.

# 7. Building and evaluating tree based algorithms

- Training various tree based algorithms (RandomForest, AdaBoost, GradientBoosting, XGBoost) using crossvalidation
- Evaluation of the models: XGBoost achieves the highest score with quite a low variance of results

# 8. Hyperparameter tuning of XGBoost

To tune XGBoost xgb.cv and sklearns XGBClassifier GridCV is used to find the best parameters. The following steps will be performed:

- Fixate the learning rate and model parameters to find the optimal n_estimators
- Using the optimal number of n_estimators max_depth and min_child_weight are tuned
- Afterwards gamma is tuned
- Then subsample and colsample are tuned
- Regarding the overfitting problem the regularization parameters are tuned
- Finally the learning rate is reduced regarding all the preliminary found hyperparameters

The various steps are evaluated regarding the F1 score metric. Using the precision/recall curve the decision threshold is optimized for the highest F1 score (precision = recall). 

The last model xgb_6 has the best scores on the test set (optimized for F1 score):

- F1 score 0.4965
- F1 score (macro avg.) 0.72
- F1 score (weighted avg.) 0.88
- Accuracy 0.88

As the last part of step 8 all features except five are dropped to train a new XGBoost model. Unfotunately the scores did not improve.

# 9. Data preparation pipelines - Neural networks

- Ordinal categorical features (OrdinalEncoder)
- Nominal categorical features (OneHotEncoder)
- Numerical features (MinMaxScaler)

Due to the decision to drop all "unknown"/NaN values imputation is not included in the pipeline.

# 10. Neural network with separated fully connected layers using the functional API of Keras

The first model is built by using the functional API. It consist of two seperated input sides that on the one hand use the input features dealing with the calls and customer specifics (input A) and on the other hand with the economic environment features (input B). After the inputs every respective side consists of multiple fully connected layers which at a given point are concatenated to be followed by multiple fully connected layers until a final output that creates the binary classification. 

The ANN with the splitted architecture is not performing better than the tuned XGBoost model xgb_6. The following list shows the comparison of the most important scores (optimized for F1 score):

- F1 score 0.485 < 0.4965 (xgb_6 best)
- F1 score (macro avg.) 0.72 < 0.72 (xgb_6 best)
- F1 score (weighted avg.) 0.88 = 0.88 (xgb_6 best)
- Accuracy 0.88 = 0.88 (xgb_6 best)

It is important to note though, that hyperparameter tuning was not possible due to the limits in terms of the functional API of Keras.

# 11. Neural network with only fully connected layers using sequential Keras API and KerasTuner

As a next step a simple fully connected neural network is created using the sequential API of Keras. For the tuning of the hyperparameters the still new KerasTuner implementation is used. The variation of parameters for the Tuner are as follows:

- Number of hidden layers: 6 - 14
- Number of neurons per hidden layer: 10 - 570, step = 70
- Activation function between hidden layers: ELU, SELU, ReLU
- The optimizers: 'adam', 'sgd', 'rmsprop', 'adadelta'

Also the fully connected neural network is not able to have better scores than the fully tuned XGBoost model xgb_6. You can see the differences in the scores in the following (Optimized for F1 score):

- F1 score 0.4852 < 0.4965 (xgb_6 best)
- F1 score (macro avg.) 0.71 < 0.72 (xgb_6 best)
- F1 score (weighted avg.) 0.88 = 0.88 (xgb_6 best)
- Accuracy 0.88 = 0.88 (xgb_6 best)

## Best model results

The best model for the given binary classification task is the XGBoost algorithm. The results are as following:

- F1 score 0.4965
- F1 score (macro avg.) 0.72
- F1 score (weighted avg.) 0.88
- Accuracy 0.88

## Authors

* **David Hedderich** - [LinkedIn](https://www.linkedin.com/in/david-hedderich-944b6886/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
Thanks for the inspiration to:
[Aurélien Geron](https://github.com/ageron/handson-ml2),

[Jason Brownlee](https://machinelearningmastery.com/)

