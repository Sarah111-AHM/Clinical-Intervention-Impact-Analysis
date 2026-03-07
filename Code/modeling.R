library(tidyverse)
library(pROC)
library(randomForest)
library(xgboost)
library(caret)
library(survival)
library(survminer)
library(lmtest)

data <- readRDS("data_clean.rds")

data_ml <- data %>%
  mutate(
    sex_num = as.numeric(sex) - 1,
    group_num = as.numeric(group) - 1,
    complication_num = as.numeric(complication) - 1
  )

cat("\n========== Multiple Linear Regression ==========\n")
lm_model <- lm(hospital_stay ~ age + sex + group, data = data)
summary(lm_model)

par(mfrow = c(2, 2))
plot(lm_model)
par(mfrow = c(1, 1))

bptest(lm_model)

cat("\n========== Logistic Regression ==========\n")
glm_model <- glm(complication ~ age + sex + group, data = data, family = binomial)
summary(glm_model)

OR_df <- exp(cbind(OR = coef(glm_model), confint(glm_model)))
print(OR_df)

pred_prob <- predict(glm_model, type = "response")
roc_curve <- roc(data$complication, pred_prob, levels = c("No", "Yes"), direction = "<")
plot(roc_curve, col = "blue", main = "ROC Curve - Logistic Model")
auc_value <- auc(roc_curve)
cat("\nAUC =", auc_value, "\n")

cat("\n========== Cox Proportional Hazards Model ==========\n")
surv_obj <- Surv(time = data$hospital_stay, event = data$complication == "Yes")
cox_model <- coxph(surv_obj ~ age + sex + group, data = data)
summary(cox_model)

fit_surv <- survfit(surv_obj ~ group, data = data)
ggsurvplot(fit_surv, data = data, pval = TRUE, risk.table = TRUE,
           title = "Survival curves without complications by group")

set.seed(123)
rf_model <- randomForest(
  complication ~ age + sex + group + hospital_stay,
  data = data_ml,
  ntree = 500,
  importance = TRUE
)
print(rf_model)

varImpPlot(rf_model, main = "Variable Importance - Random Forest")

rf_pred <- predict(rf_model, data_ml)
confusionMatrix(rf_pred, data_ml$complication)

train_matrix <- xgb.DMatrix(
  data = as.matrix(data_ml[, c("age", "sex_num", "group_num", "hospital_stay")]),
  label = data_ml$complication_num
)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  nthread = 2
)

set.seed(123)
xgb_model <- xgb.train(
  params,
  train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix),
  early_stopping_rounds = 10,
  verbose = 0
)

importance_matrix <- xgb.importance(
  model = xgb_model,
  feature_names = colnames(data_ml[, c("age", "sex_num", "group_num", "hospital_stay")])
)

xgb.plot.importance(importance_matrix, main = "Variable Importance - XGBoost")

xgb_pred_prob <- predict(xgb_model, train_matrix)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))
confusionMatrix(xgb_pred, data_ml$complication)

roc_xgb <- roc(data_ml$complication_num, xgb_pred_prob)
plot(roc_xgb, col = "red", add = TRUE)

saveRDS(lm_model, "output/tables/lm_model.rds")
saveRDS(glm_model, "output/tables/glm_model.rds")
saveRDS(rf_model, "output/tables/rf_model.rds")
saveRDS(xgb_model, "output/tables/xgb_model.rds")

cat("\nModels completed.\n")
