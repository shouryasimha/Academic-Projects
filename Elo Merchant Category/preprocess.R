install.packages("taRifx")
install.packages("elasticnet")
install.packages("plotly")
install.packages("skimr")
library(skimr)
library(taRifx)
library(dplyr)
library(AppliedPredictiveModeling)
library(caret)
library(corrplot)
library(elasticnet)
library(plotly)
setwd("C:\\Users\\merup\\Downloads\\elo\\")
trainD <- read.csv("train.csv")
merchants <- read.csv("merchants.csv")
new_merchant_transactions <- read.csv("new_merchant_transactions.csv")
historical_transactions <- read.csv("historical_transactions.csv")
testD <- read.csv("test.csv")


#elapsed_months <- function(end_date, start_date) {
#  ed <- as.POSIXlt(end_date)
#  sd <- as.POSIXlt(start_date)
#  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
#}

## convert YYYY-MM to date by
#### 1. adding -01 as day
#### 2. taking difference in months from current sys date
#trainD$duration_active <- elapsed_months(Sys.time(), 
#                                        as.Date(
#                                          paste0(trainD$first_active_month, '-01'), 
#                                          format='%Y-%m-%d')
#                                        )
#testD$duration_active <- elapsed_months(Sys.time(), 
#                                         as.Date(
#                                           paste0(testD$first_active_month, '-01'), 
#                                           format='%Y-%m-%d')
#                                        )

#str(trainD)

## create premium merchants list
#### 1. choose top 10 merchants with 12 average lag
#### 2. choose top 10 merchants with 6 average lag
#### 3. choose top 10 merchants with 3 average lag

merch_top10_avg_purch_lag12 <- merchants %>% 
  filter(!is.na(avg_purchases_lag12) & !is.infinite(avg_purchases_lag12)) %>% 
  arrange(desc(avg_purchases_lag12)) %>% 
  top_n(1000, wt =avg_purchases_lag12)
merch_top10_avg_purch_lag6 <- merchants %>% 
  filter(!is.na(avg_purchases_lag6) & !is.infinite(avg_purchases_lag6)) %>% 
  arrange(desc(avg_purchases_lag6)) %>% 
  top_n(1000, wt =avg_purchases_lag6 ) 
merch_top10_avg_purch_lag3 <- merchants %>% 
  filter(!is.na(avg_purchases_lag3) & !is.infinite(avg_purchases_lag3)) %>% 
  arrange(desc(avg_purchases_lag3)) %>% 
  top_n(1000, wt =avg_purchases_lag3 ) 

prem_merch_array <- remove.factors(merch_top10_avg_purch_lag3)$merchant_id
prem_merch_array <- append(prem_merch_array, remove.factors(merch_top10_avg_purch_lag6)$merchant_id)
prem_merch_array <- append(prem_merch_array, remove.factors(merch_top10_avg_purch_lag6)$merchant_id)
historical_transactions$prem_merch <- ifelse(historical_transactions$merchant_id %in% prem_merch_array, 1, 0)
new_merchant_transactions$prem_merch <- ifelse(new_merchant_transactions$merchant_id %in% prem_merch_array, 1, 0)

# convert factor to numeric
historical_transactions$category_1 <- as.numeric(historical_transactions$category_1)
historical_transactions$category_2 <- as.numeric(historical_transactions$category_2)
historical_transactions$category_3 <- as.numeric(historical_transactions$category_3)
new_merchant_transactions$category_1 <- as.numeric(new_merchant_transactions$category_1)
new_merchant_transactions$category_2 <- as.numeric(new_merchant_transactions$category_2)
new_merchant_transactions$category_3 <- as.numeric(new_merchant_transactions$category_3)


hist_summ_df <- 
  historical_transactions %>% 
  group_by(card_id) %>% 
  summarise(
    hist_purchase_avg = mean(purchase_amount, na.rm=TRUE), 
    hist_purchase_frequency = n(),
    hist_purchase_med = median(purchase_amount, na.rm=TRUE),
    hist_cat1_count = sum(category_1, na.rm=TRUE),
    hist_cat2_count = sum(category_2, na.rm=TRUE),
    hist_cat3_count = sum(category_3, na.rm=TRUE),
    hist_prem_merch_count = sum(prem_merch, na.rm=TRUE)
    )

newm_summ_df <- 
  new_merchant_transactions %>% 
  group_by(card_id) %>% 
  summarise(
    newm_purchase_avg = mean(purchase_amount, na.rm=TRUE), 
    newm_purchase_frequency = n(),
    newm_purchase_med = median(purchase_amount, na.rm=TRUE),
    newm_cat1_count = sum(category_1, na.rm=TRUE),
    newm_cat2_count = sum(category_2, na.rm=TRUE),
    newm_cat3_count = sum(category_3, na.rm=TRUE),
    newm_prem_merch_count = sum(prem_merch, na.rm=TRUE)
  )


## Merge trainD, hist_summ_df and newm_summ_df
gen_feat <- merge(hist_summ_df, newm_summ_df, by = "card_id", all.x = TRUE)
train_dataset <- merge(trainD, gen_feat, by = "card_id", all.x = TRUE)
trainPreProcValues  <- preProcess(train_dataset, method = c("knnImpute"))
imputed_train <- predict(trainPreProcValues, train_dataset)
write.csv(file="train_dataset.csv", x=imputed_train)

library(tidyr)
## Merge trainD, hist_summ_df and newm_summ_df
test_dataset <- merge(testD,gen_feat, by = "card_id", all.x = TRUE)
testPreProcValues  <- preProcess(test_dataset, method = c("knnImpute"))
imputed_test <- predict(testPreProcValues, test_dataset)

write.csv(file="test_dataset.csv", x=imputed_test)
## end of preprocess ##############
