---
title: "Caret Examples"
author: "davegoblue"
date: "April 14, 2016"
output: html_document
---



## Overall Objective  

This R Markdown is to document some key ideas I have seen about the "caret" library in a JHU Coursera module on Practical Machine Learning.  The basic syntax of caret ports across many different analysis techniques, and it seems extremely handy for many phases of analysis.  
  
The broad themes of caret include:  
  
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
```

```
## Warning: package 'caret' was built under R version 3.2.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Want to understand how all the pieces fit together? Buy the
## ggplot2 book: http://ggplot2.org/book/
```

```r
library(kernlab); 
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
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
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.9187823  0.8291534  0.01134362   0.02224423
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
##        -1.711e+00         -2.506e-01         -1.515e-01  
##               all              num3d                our  
##         1.182e-01          1.839e+00          5.768e-01  
##              over             remove           internet  
##         5.921e-01          2.873e+00          5.254e-01  
##             order               mail            receive  
##         4.243e-01          1.230e-01         -5.866e-01  
##              will             people             report  
##        -1.639e-01         -1.527e-01          9.342e-02  
##         addresses               free           business  
##         1.081e+00          9.776e-01          9.700e-01  
##             email                you             credit  
##         1.470e-01          8.642e-02          8.641e-01  
##              your               font             num000  
##         2.505e-01          1.018e-01          3.038e+00  
##             money                 hp                hpl  
##         2.140e-01         -1.782e+00         -1.056e+00  
##            george             num650                lab  
##        -6.767e+00          4.411e-01         -3.007e+00  
##              labs             telnet             num857  
##        -7.969e-02         -1.383e-01          2.050e+00  
##              data             num415              num85  
##        -6.739e-01          5.094e-01         -2.151e+00  
##        technology            num1999              parts  
##         9.915e-01         -4.035e-01          1.061e+00  
##                pm             direct                 cs  
##        -8.186e-01         -4.554e-01         -5.681e+02  
##           meeting           original            project  
##        -4.283e+00         -1.071e+00         -2.025e+00  
##                re                edu              table  
##        -8.848e-01         -1.253e+00         -2.463e+00  
##        conference      charSemicolon   charRoundbracket  
##        -3.916e+00         -1.220e+00         -1.430e-01  
## charSquarebracket    charExclamation         charDollar  
##        -1.253e+00          2.621e-01          4.770e+00  
##          charHash         capitalAve        capitalLong  
##         2.121e+00          1.380e-01          3.612e-03  
##      capitalTotal  
##         1.138e-03  
## 
## Degrees of Freedom: 3450 Total (i.e. Null);  3393 Residual
## Null Deviance:	    4628 
## Residual Deviance: 1379 	AIC: 1495
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
##    nonspam     662   45
##    spam         35  408
##                                           
##                Accuracy : 0.9304          
##                  95% CI : (0.9142, 0.9445)
##     No Information Rate : 0.6061          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8537          
##  Mcnemar's Test P-Value : 0.3143          
##                                           
##             Sensitivity : 0.9498          
##             Specificity : 0.9007          
##          Pos Pred Value : 0.9364          
##          Neg Pred Value : 0.9210          
##              Prevalence : 0.6061          
##          Detection Rate : 0.5757          
##    Detection Prevalence : 0.6148          
##       Balanced Accuracy : 0.9252          
##                                           
##        'Positive' Class : nonspam         
## 
```
  
There are frequently warnings thrown back by the caret library, though they often do not seem to impact the predictive ability.  The library is sometimes a bit of a black-box, and prediction is an area that often contains extreme trade-offs between parsimony, intepretability, scalability, predictive power, and the like.  It is wise to be sure the approach and final model align reasonably with how the specific end-user for the algorithm might prioritize these aims.  
  
####_Parallel Processing_  
Len Greski wrote an excellent article on using parallel processing for the train() function in caret.  This is particularly valuable for computationally intensive approaches as it increases R's ability to use my CPU from ~25% to ~75%.  See <https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md>  

The below is cut/pasted from Len's blog so that I can easily access it even when I am offine.  As setup, Len uses the "sonar" data from the "mlbench" library.  See below:  

```r
library(mlbench)
```

```
## Warning: package 'mlbench' was built under R version 3.2.4
```

```r
data(Sonar)

inTraining <- createDataPartition(Sonar$Class, p = .75, list=FALSE)
training <- Sonar[inTraining,]
testing <- Sonar[-inTraining,]

# set up training run for x / y syntax because model format performs poorly
x <- training[,-61]
y <- training[,61]
```
  
Next, Len calls libraries for parallel processing and sets trainControl to allow for parallel processing.  

```r
library(parallel)
library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.2.4
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 3.2.4
```

```
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
```
  
Then, Len runs train() function, calling fitControl to make sure it runs in parallel:  

```r
fit <- train(x, y, method="rf", data=Sonar, trControl = fitControl)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.4
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```
  
Lastly, the cluster is explicitly shut down:  

```r
stopCluster(cluster)

## Seems to be needed, at least for my machine
registerDoSEQ() ## per http://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster
```
  
Len ran several experiments on a larger dataset and found that parallel processing on an HP Omen improved run time for a Random Forest called by train() from ~450 seconds to ~200 seconds.  This looks to be very handy.  
  

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

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)

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
There are many options that can be used while training the data.  Typically, it is fine to just use the defaults, but some of the options include:  

```r
args(train.default)
```

```
## function (x, y, method = "rf", preProcess = NULL, ..., weights = NULL, 
##     metric = ifelse(is.factor(y), "Accuracy", "RMSE"), maximize = ifelse(metric %in% 
##         c("RMSE", "logLoss"), FALSE, TRUE), trControl = trainControl(), 
##     tuneGrid = NULL, tuneLength = 3) 
## NULL
```

```r
## metric can be RMSE/RSquared for continuous and Accuracy/Kappa for categorical
## weights would be if you have (for example) and unbalanced training set

args(trainControl)
```

```
## function (method = "boot", number = ifelse(grepl("cv", method), 
##     10, 25), repeats = ifelse(grepl("cv", method), 1, number), 
##     p = 0.75, search = "grid", initialWindow = NULL, horizon = 1, 
##     fixedWindow = TRUE, verboseIter = FALSE, returnData = TRUE, 
##     returnResamp = "final", savePredictions = FALSE, classProbs = FALSE, 
##     summaryFunction = defaultSummary, selectionFunction = "best", 
##     preProcOptions = list(thresh = 0.95, ICAcomp = 3, k = 5), 
##     sampling = NULL, index = NULL, indexOut = NULL, timingSamps = 0, 
##     predictionBounds = rep(FALSE, 2), seeds = NA, adaptive = list(min = 5, 
##         alpha = 0.05, method = "gls", complete = TRUE), trim = FALSE, 
##     allowParallel = TRUE) 
## NULL
```
  
Some of the key options for trainControl can include:  
  
* method= (boot, boot632, cv, repeatedcv, LOOCV) for (boostrap, bootstrap with adjustment, cross-validate, repeatedly cross-validate, leave-one-out-cross-validation)  
* number= (for boot/cross-validation, number of samples to take)  
* repeats= (number of times to repeat sub-sampling; if big, this can slow things down)  
  
It is generally valuable to set a seed, either overall or for each resample (especially for parallel fits).  
  
####_Plotting predictors_  
Plotting the predictors can be a helpful component of exploratory data analysis prior to running prediction algorithms.  This example will focus on the "Wage" data from the ISLR library.  
  

```r
library(ISLR); library(ggplot2); library(caret)
```

```
## Warning: package 'ISLR' was built under R version 3.2.4
```

```r
data(Wage)
summary(Wage)
```

```
##       year           age               sex                    maritl    
##  Min.   :2003   Min.   :18.00   1. Male  :3000   1. Never Married: 648  
##  1st Qu.:2004   1st Qu.:33.75   2. Female:   0   2. Married      :2074  
##  Median :2006   Median :42.00                    3. Widowed      :  19  
##  Mean   :2006   Mean   :42.41                    4. Divorced     : 204  
##  3rd Qu.:2008   3rd Qu.:51.00                    5. Separated    :  55  
##  Max.   :2009   Max.   :80.00                                           
##                                                                         
##        race                   education                     region    
##  1. White:2480   1. < HS Grad      :268   2. Middle Atlantic   :3000  
##  2. Black: 293   2. HS Grad        :971   1. New England       :   0  
##  3. Asian: 190   3. Some College   :650   3. East North Central:   0  
##  4. Other:  37   4. College Grad   :685   4. West North Central:   0  
##                  5. Advanced Degree:426   5. South Atlantic    :   0  
##                                           6. East South Central:   0  
##                                           (Other)              :   0  
##            jobclass               health      health_ins      logwage     
##  1. Industrial :1544   1. <=Good     : 858   1. Yes:2083   Min.   :3.000  
##  2. Information:1456   2. >=Very Good:2142   2. No : 917   1st Qu.:4.447  
##                                                            Median :4.653  
##                                                            Mean   :4.654  
##                                                            3rd Qu.:4.857  
##                                                            Max.   :5.763  
##                                                                           
##       wage       
##  Min.   : 20.09  
##  1st Qu.: 85.38  
##  Median :104.92  
##  Mean   :111.70  
##  3rd Qu.:128.68  
##  Max.   :318.34  
## 
```

```r
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
dim(training); dim(testing)
```

```
## [1] 2102   12
```

```
## [1] 898  12
```

One option is to create a feature plot for a few key variables.  

```r
featurePlot(x=training[ , c("age", "education", "jobclass")], y=training$wage, plot="pairs")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)
  
Another option is to use ggplot2 for color and regression smoothning.  

```r
## Color by jobclass
qplot(age, wage, data=training, color=jobclass)
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png)

```r
## Color by education and add a regression smooth
qq <- qplot(age, wage, data=training, color=education)
qq + geom_smooth(method="lm", formula=y~x)
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-2.png)

Another option is to create factors and then use jitter to make the boxplot.  

```r
library(Hmisc)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: Formula
```

```
## 
## Attaching package: 'Hmisc'
```

```
## The following object is masked from 'package:randomForest':
## 
##     combine
```

```
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
cutWage <- cut2(training$wage, g=3)
table(cutWage)
```

```
## cutWage
## [ 20.1, 91.7) [ 91.7,118.9) [118.9,318.3] 
##           703           721           678
```

```r
p2 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot", "jitter"))
p2
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)

```r
t1 <- table(cutWage, training$jobclass)
t1
```

```
##                
## cutWage         1. Industrial 2. Information
##   [ 20.1, 91.7)           437            266
##   [ 91.7,118.9)           373            348
##   [118.9,318.3]           274            404
```

```r
prop.table(t1, 1)
```

```
##                
## cutWage         1. Industrial 2. Information
##   [ 20.1, 91.7)     0.6216216      0.3783784
##   [ 91.7,118.9)     0.5173370      0.4826630
##   [118.9,318.3]     0.4041298      0.5958702
```
  
And yet another option is to look at the density plots.  

```r
qplot(wage, color=education, data=training, geom="density")
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

Some additional notes relevant to the prediction process include:  
  
1.  Only use the training data for plotting - no cheating with a peek at the test data!  
2.  Look for imbalances in outcomes/predictors, outliers, groups of "unexplained" points, skewed variables, etc.  
3.  Try to find the ggplot2 tutorial and the caret tutorial  
  
####_Pre-processing_  
Exploratory data analysis may reveal issues requiring data pre-processing.  This can be particularly the case with parametric approaches where skew, collinearity, missing neighbors (NA) and the like can cause problems.  
  
The spam dataset can again be analyzed for an example.  Note the extremely significant skew in the runs of capital letters (capitalAve).  
  

```r
library(caret); library(kernlab); data(spam)

inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]

hist(training$capitalAve, main="", xlab="Ave. capital run length")
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

```r
mean(training$capitalAve) ; sd(training$capitalAve)
```

```
## [1] 5.288555
```

```
## [1] 32.53952
```

One option is to use base R to standardize each of the variables to N(0,1).  The mean and sd from the training set need to be applied to the test set data also, so that we stay uninformed as to the test set data. 


```r
## Calculate metrics from training set and apply to training set
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(trainCapAveS) ; sd(trainCapAveS)
```

```
## [1] 9.100406e-18
```

```
## [1] 1
```

```r
## Apply the same transformations to the testing set
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(testCapAveS) ; sd(testCapAveS)
```

```
## [1] -0.01193141
```

```
## [1] 0.8966113
```

Alternately, the preProcess() command can be used to automatically take the same commands and apply them to the training and testing data.  
  

```r
## Set up a preProcess() command
preObj <- preProcess(training[,-58], method=c("center", "scale")) ## standardizes all variables

## Apply to training
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
mean(trainCapAveS); sd(trainCapAveS)
```

```
## [1] 9.268499e-18
```

```
## [1] 1
```

```r
## Apply to testing
testCapAveS <- predict(preObj, testing[,-58])$capitalAve
mean(testCapAveS); sd(testCapAveS)
```

```
## [1] -0.01193141
```

```
## [1] 0.8966113
```

The preProcess commands can also be passed to the train() commands.  
  

```r
set.seed(32343)
modelFit <- train(type ~ ., data=training, preProcess=c("center", "scale"), method="glm")
```

```
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

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
print(modelFit)
```

```
## Generalized Linear Model 
## 
## 3451 samples
##   57 predictor
##    2 classes: 'nonspam', 'spam' 
## 
## Pre-processing: centered (57), scaled (57) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 3451, 3451, 3451, 3451, 3451, 3451, ... 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.9179809  0.8278154  0.01824708   0.03536902
## 
## 
```

The Box-Cox transform is available, though since it is continuous, it does not solve the "many zeroes" problem, as shown by the decidedly non-linear QQ plot.  
  

```r
preObj <- preProcess(training[,-58], method=c("BoxCox")) ## standardizes all variables with Box-Cox
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2)) ; hist(trainCapAveS) ; qqnorm(trainCapAveS) ; par(mfrow=c(1,1))
```

![plot of chunk unnamed-chunk-18](figure/unnamed-chunk-18-1.png)

Since many prediction algorithms do not work well with NA, the k-nearest-neighbors approach can be applied to impute those:  
  

```r
## Create some fake NA data
set.seed(13343)
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA] <- NA

## Impute using knn
preObj <- preProcess(training[,-58], method="knnImpute")
capAve <- predict(preObj, training[,-58])$capAve
```

The preProcess() command is especially useful since it can be repeatedly applied to data as needed - train, test, validate, etc.  
  
####_Covariate creaion_  
Covariates (aka features or predictors) are the variables to be used in the model.  There are two levels:  
  
* Level 1 - from raw data to covariate (such as converting an e-mail to multiple descriptive statistics that could be used to predict whether it is spam)  
* Level 2 - transforming to create tidy covariates (such as squaring or taking a log or the like)  
  
Level 1 extractions tend to be domain specific, and the "science" (research) component is especially important during this phase.  Some guiding principles apply:  
  
* The balancing act is summarization vs. information loss  
* Examples include: 
  1.  Text Files: Frequency of words, frequency of phrases (google "ngrams"), proportion of capital letters, etc.  
  2.  Images: edges, corners, blobs, ridges ("computer vision feature detection")  
  3.  Webpages: Number/type of elements, colors, videos ("A/B testing")  
  4.  People: Height, weight, hair color, sex, country of origin  
* The more you know about the domain, the better job you will do  
* When in doubt, err on the side of more features  
* This phase can be automated, but be very careful!  There is high risk of over-fitting, finding features that work great on the training data and not at all on the testing data  
  
Level 2 conversions make tidy covariates (Level 1) in to transformed covariates (Level 2):  
  
* This is more often for some methods (regression, svm) than others (trees, forests)  
* Should only be run on the training data  
* Exploratory data analysis (plotting, tables, etc.) is the core component  
* New covariates should be added to data frames, and with recognizable names  
  
The wage data (ISLR) will again be used for an example:  

```r
library(ISLR) ; library(caret) ; data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
```
  
One common approach is to create indicator (dummy) variables:  

```r
table(training$jobclass)
```

```
## 
##  1. Industrial 2. Information 
##           1051           1051
```

```r
dummies <- dummyVars(wage ~ jobclass, data=training) ## dummyVars is a caret function
head(predict(dummies, newdata=training)) ## using predict() on dummies creates the dummy variables
```

```
##        jobclass.1. Industrial jobclass.2. Information
## 86582                       0                       1
## 161300                      1                       0
## 155159                      0                       1
## 11443                       0                       1
## 376662                      0                       1
## 450601                      1                       0
```
  
There is a function for identifying variables with near zero variance.  This is handy, as these will have no predictive power:  

```r
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv
```

```
##            freqRatio percentUnique zeroVar   nzv
## year        1.037356    0.33301618   FALSE FALSE
## age         1.027027    2.85442436   FALSE FALSE
## sex         0.000000    0.04757374    TRUE  TRUE
## maritl      3.272931    0.23786870   FALSE FALSE
## race        8.938776    0.19029496   FALSE FALSE
## education   1.389002    0.23786870   FALSE FALSE
## region      0.000000    0.04757374    TRUE  TRUE
## jobclass    1.000000    0.09514748   FALSE FALSE
## health      2.468647    0.09514748   FALSE FALSE
## health_ins  2.352472    0.09514748   FALSE FALSE
## logwage     1.061728   19.17221694   FALSE FALSE
## wage        1.061728   19.17221694   FALSE FALSE
```
  
Additionally, splines can be created on the training data and then applied to the test data:  

```r
library(splines)
bsBasis <- bs(training$age, df=3) ## Creates a polynomial variable
head(bsBasis) ## column 1 is age, column 2 is age^2, column 3 is age^3
```

```
##              1          2           3
## [1,] 0.2368501 0.02537679 0.000906314
## [2,] 0.4163380 0.32117502 0.082587862
## [3,] 0.4308138 0.29109043 0.065560908
## [4,] 0.3625256 0.38669397 0.137491189
## [5,] 0.3063341 0.42415495 0.195763821
## [6,] 0.4241549 0.30633413 0.073747105
```

```r
## Plot the results from the cubic spline
lm1 <- lm(wage ~ bsBasis, data=training)
plot(training$age, training$wage, pch=19, cex=0.75)
points(training$age, predict(lm1, newdata=training), col="red", cex=1.5, pch=19)
```

![plot of chunk unnamed-chunk-23](figure/unnamed-chunk-23-1.png)

```r
## Application of the spline to the test data
head(predict(bsBasis, age=testing$age)) ## need to use the exact same procedure
```

```
##              1          2           3
## [1,] 0.2368501 0.02537679 0.000906314
## [2,] 0.4163380 0.32117502 0.082587862
## [3,] 0.4308138 0.29109043 0.065560908
## [4,] 0.3625256 0.38669397 0.137491189
## [5,] 0.3063341 0.42415495 0.195763821
## [6,] 0.4241549 0.30633413 0.073747105
```

There is a guide on preprcessing with caret that may be valuable to look at.  Good science and domain knowledge are the keys - google "feature extraction for [good search term]" before starting in a new area.  And, leverage good exploratory analysis techniques (training set only) and then the preProcess() function in caret.  Be very careful not to overfit, and maintain a clean "test" data set to understand out of model error.  
  
####_Pre-processing with principal component analysis (PCA)_  
