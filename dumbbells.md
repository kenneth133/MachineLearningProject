# Analysis of Weight Lifting Exercises
Kenneth Lee  
March 13, 2016  

## Summary

In this report, we explore the use of several machine learning techniques in the caret package.  The dataset we will use is the Weight Lifting Exercises Dataset.  Additional information on the dataset can be found <a href="http://groupware.les.inf.puc-rio.br/har">here</a>.  The aim of the dataset is to build a prediction model to determine common mistakes made while doing unilateral dumbbell bicep curls.

## Loading Data and Feature Extraction

We begin by loading the both the training and test data.  The first 7 columns contain data on the subjects performing the exercise and data around how the data was collecting.  We remove these columns as they are not needed to build a prediction model.  Then we convert all columns to numeric except for the outcome variable.

Next, we check the test data for variables that either have no variance (all values are the same) or are missing (all rows have <NA> values).  We remove these columns from both the test and training datasets.  Lastly, we divide the training dataset into validation set (25%) and training set (75%).


```r
library(caret)

#check if file already exists in working directory
if(!file.exists("pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                  "pml-training.csv") }
if(!file.exists("pml-testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                  "pml-testing.csv") }

#read files
trainData <- read.csv(("pml-training.csv"), na.strings=c("","NA"), stringsAsFactors=FALSE)
testing <- read.csv(("pml-testing.csv"), na.strings=c("","NA"), stringsAsFactors=FALSE)

dim(trainData); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

```r
#drop columns 1:7, meta data on subjects and data collection, not needed as predictors
trainData <- trainData[,-c(1:7)]
testing <- testing[,-c(1:7)]

#update outcome variable to factor
trainData$classe <- as.factor(trainData$classe)

#convert empty strings and "NA" strings to <NA> values
trainData[,1:152] <- sapply(trainData[,1:152], as.numeric)
testing[,1:152] <- sapply(testing[,1:152], as.numeric)

#remove predictors with no variance
i <- nearZeroVar(testing,saveMetrics=TRUE)[[3]]
testing <- testing[,!i]
trainData <- trainData[,!i]

#split trainData to training set and validation set
i <- createDataPartition(y=trainData$classe, p=0.75, list=FALSE)
training <- trainData[i,]
validation <- trainData[-i,]

dim(training); dim(validation); dim(testing)
```

```
## [1] 14718    53
```

```
## [1] 4904   53
```

```
## [1] 20 53
```

## Model Selection

In this section, we test the accuracy of Linear Discriminate Analysis with and without Principal Components preprocessing and also classification trees.


```r
# Model-Based Predictions
# =======================
set.seed(8)
#linear discriminate analysis w/ principal component analysis
fit1 <- train(classe~., method="lda", preProcess="pca", data=training)
```

```
## Loading required package: MASS
```

```r
#in sample accuracy
confusionMatrix(reference=training$classe, predict(fit1, training))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.5272455
```

```r
#out of sample accuracy
confusionMatrix(reference=validation$classe, predict(fit1, validation))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.5354812
```

```r
set.seed(8)
#linear discriminate analysis w/o principal component analysis
fit2 <- train(classe~., method="lda", data=training)
#in sample accuracy
confusionMatrix(reference=training$classe, predict(fit2, training))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.7017258
```

```r
#out of sample accuracy
confusionMatrix(reference=validation$classe, predict(fit2, validation))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.7128874
```

```r
# Tree-Based Predictions
# ======================
set.seed(8)
#trees w/ k-means clustering
fit3 <- train(classe~., method="rpart", data=training)
```

```
## Loading required package: rpart
```

```r
#in sample accuracy
confusionMatrix(reference=training$classe, predict(fit3, training))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.4953119
```

```r
#out of sample accuracy
confusionMatrix(reference=validation$classe, predict(fit3, validation))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.4979608
```

Among these models, we see that LDA without Princpal Component preprocessing provides the best accuracy.  We apply this model to the test data.


```r
predict(fit2, testing)
```

```
##  [1] B A B C A C D D A A D A B A E A A B B B
## Levels: A B C D E
```

## Conclusion

The caret package offers numerous methods to build prediction models.  It is disappointing that I was only able to reach ~70% accuracy with the models above.  Due to time constraints, I was not able to include random forest, boosting, or support vector machine in this first iteration of this report.
