---
title: "Caret Examples"
author: "davegoblue"
date: "April 14, 2016"
output: html_document
---



## Overall Objective  

This R Markdown is to document some key ideas I have seen about the "caret" library in a JHU Coursera module on Practical Machine Learning.  The basic syntax of caret ports across many different analysis techniques, and it seems extremely handy for many phases of analysis.  
  
The broad themes of care include:  
  
* Pre-processing data - preProcess()  
* Partitioning raw data in to test/train - createDataPartition(), createResample(), createTimeSlices()  
* Training models on the training data - train()  
* Applying a training model to an analogous dataset - predict()  
* Comparing predictions (often to reality) - confusionMatrix()  
  
## Caret Usage  
####_Basic Example_  
Below is a basic example for using caret to make and test a simple prediction.  The "spam" dataset from library "kernlab" is assessed through this simple example.  

First, the relevant libraries are loaded, with a data partition index creates and then applied to create testing and training data.  Note that -inTrain means exclude the indices contained by inTrain, so testing and training will be mutually exclusive in this case.  


```r
library(caret); 
library(kernlab); 
data(spam, package="kernlab") ## make sure to get the kernlab version rather than ElemStatLearn

## Use createDataPartition() to create an index with 75% of the row numbers
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]
dim(training)
```

```
## [1] 3451   58
```

Next, we use caret to run GLM on the training data.  The training syntax almost always looks just like this, although with changes to the ~ formula and/or method statement.  Depending on the method (e.g., bootstrap or resample), there is sometimes an RNG component and thus merit to setting the seed.  Note that caret and all of its sub-components are frequently being updated (sometimes with impact on RNG), so there is no real guarantee of reproducibility once any component(s) of the version originally used has been sunsetted.  


```r
set.seed(32343)

## Runs the GLM and stores the outputs as modelFit
modelFit <- train(type ~ ., data=training, method="glm")
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: glm.fit: algorithm did not converge
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: glm.fit: algorithm did not converge
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
## Prints descriptors of modelFit
modelFit
```

```
## Generalized Linear Model 
## 
## 3451 samples
##   57 predictor
##    2 classes: 'nonspam', 'spam' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 3451, 3451, 3451, 3451, 3451, 3451, ... 
## Resampling results
## 
##   Accuracy  Kappa      Accuracy SD  Kappa SD  
##   0.917132  0.8263397  0.01646928   0.03148507
## 
## 
```

```r
## Prints coefficients that come back from the final GLM selected
modelFit$finalModel
```

```
## 
## Call:  NULL
## 
## Coefficients:
##       (Intercept)               make            address  
##        -1.521e+00         -4.595e-01         -1.462e-01  
##               all              num3d                our  
##         2.402e-01          2.683e+00          5.731e-01  
##              over             remove           internet  
##         1.075e+00          2.166e+00          1.014e+00  
##             order               mail            receive  
##         6.769e-01          1.433e-01         -3.651e-02  
##              will             people             report  
##        -2.165e-01         -1.718e-01          1.545e-01  
##         addresses               free           business  
##         6.325e-01          1.070e+00          1.179e+00  
##             email                you             credit  
##         1.997e-01          1.128e-01          1.539e+00  
##              your               font             num000  
##         2.199e-01          1.811e-01          1.982e+00  
##             money                 hp                hpl  
##         3.484e-01         -2.662e+00         -6.094e-01  
##            george             num650                lab  
##        -9.597e+00          3.763e-01         -2.291e+00  
##              labs             telnet             num857  
##        -5.698e-01          1.112e+00          2.479e+00  
##              data             num415              num85  
##        -6.901e-01         -1.231e+01         -1.621e+00  
##        technology            num1999              parts  
##         1.070e+00          1.651e-01          1.234e+00  
##                pm             direct                 cs  
##        -8.523e-01         -4.626e-01         -5.432e+02  
##           meeting           original            project  
##        -3.560e+00         -1.434e+00         -2.076e+00  
##                re                edu              table  
##        -9.890e-01         -1.644e+00         -2.836e+00  
##        conference      charSemicolon   charRoundbracket  
##        -3.298e+00         -1.069e+00         -2.429e-01  
## charSquarebracket    charExclamation         charDollar  
##        -5.441e-01          2.625e-01          5.190e+00  
##          charHash         capitalAve        capitalLong  
##         2.238e+00          4.304e-03          8.004e-03  
##      capitalTotal  
##         6.973e-04  
## 
## Degrees of Freedom: 3450 Total (i.e. Null);  3393 Residual
## Null Deviance:	    4628 
## Residual Deviance: 1333 	AIC: 1449
```

```r
## Applies modelFit (which was built from training) to make predictions on the testing data
predictions <- predict(modelFit, newdata=testing)

## Runs the confusion matrix and outputs some interesting statistics
confusionMatrix(predictions, testing$type)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction nonspam spam
##    nonspam     661   48
##    spam         36  405
##                                           
##                Accuracy : 0.927           
##                  95% CI : (0.9104, 0.9413)
##     No Information Rate : 0.6061          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8463          
##  Mcnemar's Test P-Value : 0.2301          
##                                           
##             Sensitivity : 0.9484          
##             Specificity : 0.8940          
##          Pos Pred Value : 0.9323          
##          Neg Pred Value : 0.9184          
##              Prevalence : 0.6061          
##          Detection Rate : 0.5748          
##    Detection Prevalence : 0.6165          
##       Balanced Accuracy : 0.9212          
##                                           
##        'Positive' Class : nonspam         
## 
```
  
There are frequently warnings thrown back by the caret library, though they often do not seem to impact the predictive ability.  The library is sometimes a bit of a black-box, and prediction is an area that often contains extreme trade-offs between parsimony, intepretability, scalability, predictive power, and the like.  It is wise to be sure the approach and final model align reasonably with how the specific end-user for the algorithm might prioritize these aims.  
  
####_Data Slicing_  
Continuing with the spam dataset, we may want to slice it.  There are three common methods described below:  
  
* k-fold  
* resample  
* time slices  
  

```r
## ?createFolds will bring up the help menu for all types of data splitting

## Document the length of the spam dataset
length(spam$type)
```

```
## [1] 4601
```

```r
## Create 10 folds (all independent)
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=FALSE)
sapply(folds, FUN=length)
```

```
## Fold01 Fold02 Fold03 Fold04 Fold05 Fold06 Fold07 Fold08 Fold09 Fold10 
##    460    461    460    459    461    459    460    460    461    460
```

```r
sum(sapply(folds, FUN=length)) ## same length as original
```

```
## [1] 4601
```

```r
myCheck <- NULL
for (intCtr in 1:10) { myCheck <- c(myCheck, folds[[intCtr]]) }
identical(1:length(spam$type), myCheck[order(myCheck)]) ## all elements used exactly once
```

```
## [1] TRUE
```

```r
folds[[1]][1:10]
```

```
##  [1] 24 27 32 40 41 43 55 58 63 68
```

```r
## Create 10 resamples (data can be used multiple times)
set.seed(32323)
folds <- createResample(y=spam$type, times=10, list=TRUE)
sapply(folds, length) ## Each fold same length as original
```

```
## Resample01 Resample02 Resample03 Resample04 Resample05 Resample06 
##       4601       4601       4601       4601       4601       4601 
## Resample07 Resample08 Resample09 Resample10 
##       4601       4601       4601       4601
```

```r
myCheck <- NULL
for (intCtr in 1:10) { myCheck <- c(myCheck, folds[[intCtr]]) }
myCount <- NULL
for (intCtr in 1:length(spam$type)) { myCount <- c(myCount, sum(myCheck==intCtr)) }
plot(x=1:length(spam$type), y=myCount, col="red", pch=19) ## Indices used different # times
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png)

```r
## Create test/train data for a time series (needs to be reasonably contiguous)
set.seed(32323)
tme <- 1:1000 ## For my series, I am interested in times 1-1000
folds <- createTimeSlices(y=tme, initialWindow=20, horizon=10) ## train on 20, test on 10
names(folds)
```

```
## [1] "train" "test"
```

```r
folds$train[[1]] ## First fold trains on 1-20
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
```

```r
folds$test[[1]] ## First fold tests on 21-30
```

```
##  [1] 21 22 23 24 25 26 27 28 29 30
```

```r
folds$train[[length(folds$train)]] ## Last fold trains on 971-990 (max is 1000 per tme)
```

```
##  [1] 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987
## [18] 988 989 990
```

```r
folds$test[[length(folds$train)]] ## Last fold tests on 991-1000 (max is 1000 per tme)
```

```
##  [1]  991  992  993  994  995  996  997  998  999 1000
```

Some of the algorithms in the caret library take care of these automatically.  However, it is nice to have the flexibility to create partitions in whatever manner is best suited for the task at hand.  
  
####_Training options_  
