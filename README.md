# Evaluating-Multi-Class-Classifiers
Content and associated scripts for a article on evaluating multi-class classifier for an internal journal 
Assessing Multi-Class Classifiers

## Introduction 
In Machine Learning, classification algorithms approximate the mapping function which classifies input instances to the target classes.   There are three main flavors of classifiers

1.	Binary – only two mutually -exclusive possible outcomes e.g. Hotdog or Not 

2.	Multi-class – many mutually -exclusive possible outcomes e.g. Animal, Vegetable, OR Mineral

3.	Multi-label – many overlapping possible outcomes – a document can have content on Sports, Finance, AND Politics    

This article will focus on the evaluation metrics for comparing multi-class classifications.

## Multi-class Classification 
Multi-class classification can in-turn be separated into three groups:
**1.	Native classifiers** 
These include familiar classifier families such as Support Vector Machines (SVM)s, Classification And Regression Trees (CART) , KNN, Naïve Bayes (NB), and Neural Nets with multi-layer output nodes.

**2.	 Multi-class wrappers on binary classifiers**
These hybrid classifiers reduce the problem to smaller chunks which can then be solved with dedicated binary classifiers. The two main variants are:
a.	One vs All (OVA) : a binary classifier tuned to each class separately identifies that class as a positive and all others as negative. 
b.	All vs All (AVA): Each binary classifier is trained to discriminate between individual pairs of classes and discard the rest. Each new data point is evaluated by the classifier and assigned the class with most votes.

**3.	Hierarchical classifiers** 
This group uses hierarchical methods to separate output space into nodes corresponding to target classes using a tree-based architecture. Useful for large class class outputs but not very common an example is here.

## Measurement metrics 
Selecting the best metrics for evaluating the performance of a given classifier on dataset is guided by a number of consideration including the class-balance and expected outcomes. One particular performance measure may evaluate a classifier from a single perspective and often fail to measure others. Consequently, there is no unified metric to select measure the generalized performance of a classifier. 

Two methods, micro-averaging and macro-averaging, are used to extract a single number for each of the precision, recall and other metrices across multiple classes. A macro-average calculates the metric autonomously for each class to calculate the average. In contrast, the micro-average calculates average metric from the aggregate contributions of all classes. Micro -average is used in unbalanced datasets as this method takes the frequency of each class into consideration. The micro average precision, recall, and accuracy scores are mathematically equivalent.

A Decision Tree Classifier (DTC), a Support Vector Machine (SVM) , a Gaussian Naïve Bayes (GNB) , and a K Nearest Neighbor  (KNN) algorithms performance on classification of the Glass Identification Dataset  from UCI ML Repository was used to calculate the following metrices. 
### Class- Centric Metrics 

These metrices provide detailed evaluation of model performance at the class level.

**Classification report** 
The classification report provides the main classification metrics on a per-class basis. 
a)	Precision (tp / (tp + fp) ) measures the ability of a classifier to identify only the correct instances for each class.
b)	Recall (tp / (tp + fn) is the ability of a classifier to find all correct instances per class.
c)	F1 score is a weighted harmonic mean of precision and recall normalized between 0 and 1. F score of 1 indicates a perfect balance as precision and the recall are inversely related. A high F1 score is useful where both high recall and precision is important.  
d)	Support is the number of actual occurrences of the class in the test data set. Imbalanced support in the training data may indicate the need for stratified sampling or rebalancing.       

**Confusion Matrix**
A confusion matrix shows the combination of the actual and predicted classes. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class. It is a good measure of wether models can account for the overlap in class properties and to understand which classes are most easily confused.

**Class Prediction Error**
This is a useful extension of the confusion matrix and visualizes the misclassified classes as a stacked bar. Each bar is a composite measure of predicted classes.
    
### Aggregate metrics 
These provide a score for the overall performance of the classifier across the class spectrum.

**Cohen’s Kappa**
This is one of the best metrics for evaluating multi-class classifiers on imbalanced datasets.
The traditional metrics from the classification report are biased towards the majority class and assumes an identical distribution of the actual and predicted classes. In contrast, Cohen’s Kappa Statistic measures the proximity of the predicted classes to the actual classes when compared to a random classification.  The output is normalized between 0 and 1 the metrics for each classifier, therefore can be directly compared  across the classification task. Generally closer the score is to one, better the classifier.

**Cross-Entropy** 
Cross entropy measures the extent to which the predicted probabilities match the given data, and is useful for probabilistic classifiers such as Naïve Bayes.  It is a more generic form of the logarithmic loss function, which was derived from neural network architecture, and is used to quantify the cost of inaccurate predictions. The classifier with the lowest log loss is preferred.

**Mathews Correlation Coefficient (MCC)**
MCC , originally devised for binary classification on unbalanced classes, has been extended to evaluates multiclass  classifiers by  computing the correlation coefficient between the observed and predicted classifications. A coefficient of +1 represents a perfect prediction, 0 is similar to a random prediction and −1 indicates an inverse prediction.

## Conclusion 
Best practice model section methodology for a multi-class classification problem is to use a basket of metrics to make to select the best algorithm depending on the nuances of expected outcomes and the nature of the data.

## References
http://atour.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
https://medium.com/datadriveninvestor/understanding-the-log-loss-function-of-xgboost-8842e99d975d
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass
http://www.scikit-yb.org/en/latest/api/classifier/index.html
https://en.wikipedia.org/wiki/Multiclass_classification
http://gabrielelanaro.github.io/blog/2016/02/03/multiclass-evaluation-measures.html
https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
