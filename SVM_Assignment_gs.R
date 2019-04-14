library(kernlab)
library(readr)
library(caret)
library(caTools)
library(ggplot2)
library(readr)
library(gridExtra)

setwd("C:/PGDDS/SVM assignment")


#Loading Data

img_trn_data<- read.csv("mnist_train.csv",sep = ",", stringsAsFactors = F,header = F)
img_test_data<-  read.csv("mnist_test.csv",sep = ",", stringsAsFactors = F,header = F)

#Understanding Dimensions

dim(img_trn_data)

#Structure of the dataset

str(img_trn_data)

#printing first few rows

head(img_trn_data)

#Exploring the data

summary(img_trn_data)

# Counting the number of records in the training data

img_trn_data_nrow<-nrow(img_trn_data)

# Checking for duplicated records
img_trn_data_unrow<-nrow(unique(img_trn_data))

if(img_trn_data_nrow==img_trn_data_unrow) {print("No duplicated Records in training data")} else {print("Duplicated records.Need Cleaning")}

# Counting the number of records in the test data

img_test_data_nrow<-nrow(img_test_data)

# Checking for duplicated records
img_test_data_unrow<-nrow(unique(img_test_data))

if(img_test_data_nrow==img_test_data_unrow) {print("No duplicated Records in test data")} else {print("Duplicated records.Need Cleaning")}


#checking missing value

sum(is.na(img_trn_data))

sum(is.na(img_test_data))

#Making our target class to factor

colnames(img_trn_data)[1] <- "Digit"
colnames(img_test_data)[1] <- "Digit"

img_trn_data$Digit<-factor(img_trn_data$Digit)
img_test_data$Digit<-factor(img_test_data$Digit)

# Split the data into train and test set

set.seed(1)
train.indices = sample(1:nrow(img_trn_data), 0.15*nrow(img_trn_data))
train = img_trn_data[train.indices, ]

summary(train)

#Using Linear Kernel
img_mdl_linear <- ksvm(Digit~ ., data = train, scale = FALSE, kernel = "vanilladot")
img_eval_linear<- predict(img_mdl_linear, img_test_data)

#confusion matrix - Linear Kernel
confusionMatrix(img_eval_linear,img_test_data$Digit)

#Confusion Matrix and Statistics

#Reference
#Prediction    0    1    2    3    4    5    6    7    8    9
#0  963    0   12    3    1   13   10    1    9    9
#1    0 1118   15    4    1    5    3   13   13    4
#2    0    2  944   34    6    5   12   19   15    3
#3    0    2   12  899    0   46    1    8   28   15
#4    1    0    9    2  918    7    7   10   12   60
#5    7    2    3   34    2  778   10    0   39    8
#6    5    4    9    2    7   15  912    0   12    0
#7    3    0    7    8    6    3    1  939    8   37
#8    1    7   19   19    0   14    2    6  828    6
#9    0    0    2    5   41    6    0   32   10  867

#Overall Statistics

#Accuracy : 0.9166         
#95% CI : (0.911, 0.9219)
#No Information Rate : 0.1135         
#P-Value [Acc > NIR] : < 2.2e-16      

#Kappa : 0.9073         
#Mcnemar's Test P-Value : NA             

#Statistics by Class:

#Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity            0.9827   0.9850   0.9147   0.8901   0.9348   0.8722   0.9520   0.9134   0.8501   0.8593
#Specificity            0.9936   0.9935   0.9893   0.9875   0.9880   0.9885   0.9940   0.9919   0.9918   0.9893
#Pos Pred Value         0.9432   0.9507   0.9077   0.8892   0.8947   0.8811   0.9441   0.9279   0.9180   0.9003
#Neg Pred Value         0.9981   0.9981   0.9902   0.9877   0.9929   0.9875   0.9949   0.9901   0.9840   0.9843
#Prevalence             0.0980   0.1135   0.1032   0.1010   0.0982   0.0892   0.0958   0.1028   0.0974   0.1009
#Detection Rate         0.0963   0.1118   0.0944   0.0899   0.0918   0.0778   0.0912   0.0939   0.0828   0.0867
#Detection Prevalence   0.1021   0.1176   0.1040   0.1011   0.1026   0.0883   0.0966   0.1012   0.0902   0.0963
#Balanced Accuracy      0.9881   0.9892   0.9520   0.9388   0.9614   0.9303   0.9730   0.9526   0.9210   0.9243
 


#Using RBF Kernel
img_mdl_RBF <- ksvm(Digit~ ., data = train, scale = FALSE, kernel = "rbfdot")
img_eval_RBF<- predict(img_mdl_RBF, img_test_data)

#confusion matrix - RBF Kernel
confusionMatrix(img_eval_RBF,img_test_data$Digit)


#Confusion Matrix and Statistics

#Reference
#Prediction    0    1    2    3    4    5    6    7    8    9
#0  968    0    8    0    1    8    8    0    5    5
#1    0 1123    1    0    0    3    3   12    1    6
#2    2    3  980   13    4    4    2   20    3    1
#3    0    2    3  963    0   19    0    1    9    5
#4    0    0    9    0  943    3    3    4    5   16
#5    4    1    1   11    0  836    9    0   14    6
#6    4    3    5    1    6    9  931    0    5    1
#7    1    1   11    7    1    2    0  966    6    7
#8    1    2   14   11    2    5    2    3  923    8
#9    0    0    0    4   25    3    0   22    3  954

#Overall Statistics

#Accuracy : 0.9587          
#95% CI : (0.9546, 0.9625)
#No Information Rate : 0.1135          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.9541          
#Mcnemar's Test P-Value : NA              

#Statistics by Class:

#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity            0.9878   0.9894   0.9496   0.9535   0.9603   0.9372   0.9718   0.9397   0.9476   0.9455
#Specificity            0.9961   0.9971   0.9942   0.9957   0.9956   0.9949   0.9962   0.9960   0.9947   0.9937
#Pos Pred Value         0.9651   0.9774   0.9496   0.9611   0.9593   0.9478   0.9648   0.9641   0.9506   0.9436
#Neg Pred Value         0.9987   0.9986   0.9942   0.9948   0.9957   0.9939   0.9970   0.9931   0.9944   0.9939
#Prevalence             0.0980   0.1135   0.1032   0.1010   0.0982   0.0892   0.0958   0.1028   0.0974   0.1009
#Detection Rate         0.0968   0.1123   0.0980   0.0963   0.0943   0.0836   0.0931   0.0966   0.0923   0.0954
#Detection Prevalence   0.1003   0.1149   0.1032   0.1002   0.0983   0.0882   0.0965   0.1002   0.0971   0.1011
#Balanced Accuracy      0.9919   0.9932   0.9719   0.9746   0.9779   0.9661   0.9840   0.9678   0.9712   0.9696


############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(Digit~ ., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)