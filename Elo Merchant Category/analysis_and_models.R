install.packages("skimr")
install.packages("taRifx")
install.packages("dplyr")
install.packages("AppliedPredictiveModeling")
install.packages("caret")
install.packages("corrplot")
install.packages("elasticnet")
install.packages("plotly")
install.packages("tidyr")
install.pacakages("purrr")
install.pacakages("ggpolot2")
library(ggplot2)
library(purrr)
library(tidyr)
library(skimr)
library(taRifx)
library(dplyr)
library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(elasticnet)
library(plotly)

setwd("C:\\Users\\merup\\Downloads\\elo\\")
train_dataset <- read.csv("train_dataset.csv")
submit_test_dataset <- read.csv("test_dataset.csv")
names(train_dataset)

library(purrr)
library(tidyr)
library(ggplot2)

train_dataset %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()



## Split data into train and test
set.seed(21)
ind = createDataPartition(train_dataset$target, p = 2/3, list = FALSE)
trainxy <- select( train_dataset[ind,], -c(X, card_id, first_active_month))
testxy <- select( train_dataset[-ind,], -c(X, card_id, first_active_month))
str(trainxy)
str(testxy)

## analysis - correlation of predictors
dev.off()
plot.new()
#trainxy <- 
M <- cor(trainxy)
corrplot(M, method="circle")

# 1. OLS all predictors
####################
sum(is.na(trainxy))

# let's try 10 fold cross validation
set.seed(21)
#xvCtl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
xvCtl <- trainControl(method = "cv", number = 10)

lmFit1 <- train(target~., data = trainxy,
                preProcess = c("BoxCox", "center", "scale"),
                 method = "lm", trControl = xvCtl)

print(lmFit1$resample)
asdf <- apply(lmFit1$resample[, 1:2], 2, sd)
testx <- select(testxy, -target)
testy <- as.numeric(unlist(select(testxy, target)))

str(submit_test_dataset)
lm_submit_test_dataset <- submit_test_dataset
lm_submit_testx <- select(lm_submit_test_dataset[], -c(X, card_id, first_active_month))
lm_submit_test_dataset['target'] <- predict(lmFit1, lm_submit_testx)
lm_submit <- select(lm_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="lm_submission.csv", x=lm_submit, row.names=FALSE)

testPred <- predict(lmFit1, testx)
resid <- testPred - testy
typeof(testPred)
plot(x=testPred, y=resid, xlim=c(-4,4), ylim=c(-10,10), main = "OLS predicted Vs Residuals")
abline(h=0, col="blue")

# 2. PLS
###############
set.seed(27)
plsFit <- train(target~., data = trainxy,
                preProcess = c("BoxCox", "center", "scale"),
                method = "pls",
                tuneLength = 25,
                trControl = xvCtl)

plot(plsFit, main="PLS")
plsFit
plot(varImp(plsFit), main = "PLS")
pls_pred <- predict(plsFit,testx)

pls_submit_test_dataset <- submit_test_dataset
submit_testx <- select(pls_submit_test_dataset[], -c(X, card_id, first_active_month))
pls_submit_test_dataset['target'] <- predict(plsFit, submit_testx)
str(pls_submit_test_dataset)
pls_submit <- select(pls_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="pls_submission.csv", x=pls_submit, row.names=FALSE)

xyplot(plsFit$results$RMSE ~ plsFit$results$ncomp, 
       type = c("p", "g"), 
       xlab = "tuneLength", 
       ylab = "RMSE", 
       main = "PLS choosing tuneLength")

# 3. Ridge
#############

ridgeGrid <- data.frame(.lambda = seq(0, 1.5e-02, length =10))
set.seed(21)
ridgeRegFit <- train(target~., data = trainxy,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = xvCtl,
                     preProcess = c("BoxCox", "center", "scale"))
RMSE(predict(ridgeRegFit, testx), testxy$target)
ridge_submit_test_dataset <- submit_test_dataset
ridge_submit_testx <- select(ridge_submit_test_dataset[], -c(X, card_id, first_active_month))
ridge_submit_test_dataset['target'] <- predict(ridgeRegFit, ridge_submit_testx)
ridge_submit <- select(ridge_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="ridge_submission.csv", x=ridge_submit, row.names=FALSE)

ridgeRegFit

# plot RMSE ~ Lambda
xyplot(ridgeRegFit$results$RMSE ~ ridgeRegFit$results$lambda, 
       type = c("l", "g"), 
       xlab = "Lambda", 
       ylab = "RMSE", 
       main = "Ridge Regression Tuning Lambda")

# 4.Lasso/elastice net
###################

enetGrid <- expand.grid(.lambda = c(0.00001, 0.0000001),
                        .fraction = seq(0.1, 1, length = 20))
set.seed(21)
enetTune <- train(target~., data = trainxy,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = xvCtl,
                  preProcess = c("BoxCox", "center", "scale"))
plot(enetTune, main='enet')
RMSE(predict(enetTune, testx), testxy$target)
enetTune
enet_submit_test_dataset <- submit_test_dataset
enet_submit_testx <- select(enet_submit_test_dataset[], -c(X, card_id, first_active_month))
enet_submit_test_dataset['target'] <- predict(enetTune, enet_submit_testx)
enet_submit <- select(enet_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="enet_submission.csv", x=enet_submit, row.names=FALSE)
enetTune$results
enet_pred <- predict(enetTune,testx)


# 5. Neural Network
set.seed(27)
nngrid <- expand.grid(.decay = c(0.5, 0.1), .size = c( 3, 4, 5, 6, 7))
nnfit <- train(target~., 
               data = trainxy, 
               method = "nnet", 
               maxit = 1000, 
               tuneGrid = nngrid,
               preProc = c("center","scale","BoxCox"),
               trace = F, 
               linout = 1) 
nnfit

nn_pred <- predict(nnfit,testx)
library(ModelMetrics)
testx <- select(testxy, -target)
testy <- select(testxy_outl_rem, target)
RMSE(predict(nnfit, testx), testxy$target)
xyplot(nn_pred ~ testxy$target, 
       type = c("p", "r"), 
       ylab = "predicted", 
       xlab = "observed", 
       main = "Neural Network Observed Vs Predicted")
plot(nnfit,main="Neural Networks")
plot(varImp(nnfit), main="Neural Networks")
nnfit
nn_submit_test_dataset <- submit_test_dataset
nn_submit_testx <- select(nn_submit_test_dataset[], -c(X, card_id, first_active_month))
nn_submit_test_dataset['target'] <- predict(nnfit, nn_submit_testx)
nn_submit <- select(nn_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="nn_submission.csv", x=nn_submit, row.names=FALSE)


# 7. random forest
set.seed(27)
str(trainxy)
mtry <- sqrt(ncol(trainxy))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(target~., 
                    data = trainxy, 
                    method='rf', 
                    metric='RMSE', 
                    tuneGrid=tunegrid, 
                    trControl=xvCtl,
                    preProc = c("center", "scale", "BoxCox"))
plot(rf_default$finalModel)
install.packages("party")
library(party)
plot(rf_default, type="simple")
fancyRpartPlot(rf_default$frame)
names(rf_default)
rf_pred <- predict(rf_default,subtestx)
xyplot(rf_pred ~ subtesty$target, 
       type = c("p", "r"), 
       ylab = "predicted", 
       xlab = "observed", 
       main = "Random Forest Observed Vs Predicted")

rf_submit_test_dataset <- submit_test_dataset
rf_submit_testx <- select(rf_submit_test_dataset[], -c(X, card_id, first_active_month))
rf_submit_test_dataset['target'] <- predict(rf_default, rf_submit_testx)
rf_submit <- select(rf_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="rf_submission.csv", x=rf_submit, row.names=FALSE)


# 8. XGBoost
install.packages("xgboost")
library(xgboost)
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=c(0.1,0.2,0.3,0.4, 0.5),
                               max_depth=c(3,4,6),
                               nrounds=100,
                               gamma=1,
                               min_child_weight=2,
                               subsample = 0.8
                              )
modelxgboost <- train(target~., 
                      data = trainxy,
                      method = "xgbTree",
                      trControl = xvCtl,
                      tuneGrid=parametersGrid,
                      preProcess = c("BoxCox", "center", "scale"))
RMSE(predict(modelxgboost, testx), testxy$target)
fit.time(modelxgboost)
modelxgboost$times
modelxgboost
plot(modelxgboost, main = "XGBoost")
xgb_submit_test_dataset <- submit_test_dataset
xgb_submit_testx <- select(xgb_submit_test_dataset[], -c(X, card_id, first_active_month))
xgb_submit_test_dataset['target'] <- predict(modelxgboost, xgb_submit_testx)
xgb_submit <- select(xgb_submit_test_dataset[], c(card_id, target)) %>% mutate_if(is.numeric , replace_na, replace = 0)
write.csv(file="xgb_submission.csv", x=xgb_submit, row.names=FALSE)

testPred <- predict(modelxgboost, testx)
resid <- testPred - testy
typeof(testPred)
plot(x=testPred, y=resid, xlim=c(-4,4), ylim=c(-10,10), main = "XGB predicted Vs Residuals")
abline(h=0, col="blue")


plot(modelxgboost)
plot(varImp(modelxgboost), main = "XGBoost")
modelxgboost
predictions<-predict(modelxgboost,testDF)
models.list <- list(lmFit1, plsFit, ridgeRegFit, enetTune)

## Summary statistics
models.list %>% 
  # get resamples results
  resamples %>% .$values %>% 
  # only select numeric columns
  select_if(is.numeric) %>% 
  # calculate standard deviation
  summarise_all(.funs = sd) 

nnfit$resample %>% select_if(is.numeric)  %>% summarise_all(.funs = sd)

coef(enetTune$finalModel)
