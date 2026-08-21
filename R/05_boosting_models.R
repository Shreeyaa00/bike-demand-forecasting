library(tidyverse)
library(caret)
library(gbm)
library(xgboost)

df <- read.csv("data/bike_clean.csv")

df <- df %>%
  mutate(
    season = factor(season),
    yr = factor(yr),
    mnth = factor(mnth),
    hr = factor(hr),
    holiday = factor(holiday),
    weekday = factor(weekday),
    workingday = factor(workingday),
    weathersit = factor(weathersit),
    date = as.Date(date)
  )

set.seed(42)

trainIndex <- createDataPartition(
  df$cnt,
  p = 0.80,
  list = FALSE
)

train <- df[trainIndex, ]
test <- df[-trainIndex, ]
cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")

train_model <- train %>%
  select(-date)

test_model <- test %>%
  select(-date)

evaluate_model <- function(actual, predicted) {
  
  rmse <- sqrt(
    mean((actual - predicted)^2)
  )
  
  mae <- mean(
    abs(actual - predicted)
  )
  
  r_squared <- 1 -
    sum((actual - predicted)^2) /
    sum((actual - mean(actual))^2)
  
  data.frame(
    RMSE = rmse,
    MAE = mae,
    R_squared = r_squared
  )
}
set.seed(42)

gbm_model <- gbm(
  cnt ~ .,
  data = train_model,
  distribution = "gaussian",
  n.trees = 500,
  interaction.depth = 5,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)


gbm_predictions <- predict(
  gbm_model,
  newdata = test_model,
  n.trees = 500
)


gbm_metrics <- evaluate_model(
  test_model$cnt,
  gbm_predictions
)

gbm_metrics$Model <- "Gradient Boosting"
cat("GRADIENT BOOSTING\n")
print(gbm_metrics)

gbm_importance <- summary(
  gbm_model,
  plotit = FALSE
)

cat("GRADIENT BOOSTING FEATURE IMPORTANCE\n")
print(gbm_importance)

x_train <- model.matrix(
  cnt ~ .,
  data = train_model
)[, -1]

x_test <- model.matrix(
  cnt ~ .,
  data = test_model
)[, -1]

y_train <- train_model$cnt
y_test <- test_model$cnt


# Check dimensions
cat("\nXGBoost training dimensions:\n")
cat("Rows:", nrow(x_train), "\n")
cat("Columns:", ncol(x_train), "\n")

set.seed(42)

xgb_model <- xgboost(
  x = x_train,
  y = y_train,
  objective = "reg:squarederror",
  nrounds = 500,
  max_depth = 6,
  learning_rate = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  eval_metric = "rmse"
)

xgb_predictions <- predict(
  xgb_model,
  x_test
)

xgb_metrics <- evaluate_model(
  y_test,
  xgb_predictions
)

xgb_metrics$Model <- "XGBoost"

cat("XGBOOST\n")
print(xgb_metrics)

xgb_importance <- xgb.importance(
  model = xgb_model
)

cat("XGBOOST FEATURE IMPORTANCE\n")
print(
  head(
    xgb_importance,
    15
  )
)

previous_models <- data.frame(
  
  Model = c(
    "Linear Regression",
    "Lasso Regression",
    "Elastic Net",
    "Ridge Regression",
    "Decision Tree",
    "Random Forest"
  ),
  
  RMSE = c(
    99.151,
    99.158,
    99.159,
    100.394,
    69.658,
    47.634
  ),
  
  MAE = c(
    73.909,
    73.876,
    73.878,
    74.734,
    47.317,
    31.720
  ),
  
  R_squared = c(
    0.6982,
    0.6982,
    0.6982,
    0.6906,
    0.8511,
    0.9304
  )
)


gbm_results <- gbm_metrics %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )


xgb_results <- xgb_metrics %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )


all_models <- bind_rows(
  previous_models,
  gbm_results,
  xgb_results
) %>%
  arrange(RMSE)

cat("FULL MODEL LEADERBOARD\n")
print(all_models)

comparison_long <- all_models %>%
  pivot_longer(
    cols = c(RMSE, MAE),
    names_to = "Metric",
    values_to = "Value"
  )


ggplot(
  comparison_long,
  aes(
    x = reorder(Model, Value),
    y = Value
  )
) +
  geom_col() +
  facet_wrap(
    ~ Metric,
    scales = "free_y"
  ) +
  coord_flip() +
  labs(
    title = "Full Model Comparison",
    x = "Model",
    y = "Error"
  ) +
  theme_minimal()

boosting_results <- bind_rows(
  gbm_results,
  xgb_results
) %>%
  arrange(RMSE)


best_boosting <- boosting_results$Model[1]


if (best_boosting == "XGBoost") {
  
  best_predictions <- xgb_predictions
  
} else {
  
  best_predictions <- gbm_predictions
  
}


actual_vs_predicted <- data.frame(
  Actual = y_test,
  Predicted = best_predictions
)


ggplot(
  actual_vs_predicted,
  aes(
    x = Actual,
    y = Predicted
  )
) +
  geom_point(alpha = 0.3) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = paste(
      "Actual vs Predicted:",
      best_boosting
    ),
    x = "Actual Demand",
    y = "Predicted Demand"
  ) +
  theme_minimal()

write.csv(
  all_models,
  "outputs/full_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  gbm_importance,
  "outputs/gradient_boosting_importance.csv",
  row.names = FALSE
)

write.csv(
  xgb_importance,
  "outputs/xgboost_importance.csv",
  row.names = FALSE
)


cat("\nResults saved to outputs/\n")
