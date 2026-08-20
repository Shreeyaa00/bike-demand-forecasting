# ============================================================
# Bike Rental Demand Forecasting
# 06 - Time-Based Validation
# ============================================================


# ============================================================
# 1. Packages
# ============================================================

library(tidyverse)
library(caret)
library(gbm)
library(randomForest)
library(xgboost)


# ============================================================
# 2. Load Clean Data
# ============================================================

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
  ) %>%
  arrange(date, as.numeric(as.character(hr)))


# ============================================================
# 3. Inspect Date Range
# ============================================================

cat("Full dataset date range:\n")
print(range(df$date))

cat("\nTotal observations:", nrow(df), "\n")


# ============================================================
# 4. Time-Based Train/Test Split
# ============================================================

# Use the latest 20% of observations as the future test set

split_index <- floor(
  0.80 * nrow(df)
)

train <- df[1:split_index, ]
test <- df[(split_index + 1):nrow(df), ]


cat("\n==============================\n")
cat("TIME-BASED SPLIT\n")
cat("==============================\n")

cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")

cat("\nTraining period:\n")
print(range(train$date))

cat("\nTesting period:\n")
print(range(test$date))


# ============================================================
# 5. Remove Date
# ============================================================

train_model <- train %>%
  select(-date)

test_model <- test %>%
  select(-date)


# ============================================================
# 6. Evaluation Function
# ============================================================

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


# ============================================================
# 7. LINEAR REGRESSION
# ============================================================

linear_model <- lm(
  cnt ~ .,
  data = train_model
)

linear_predictions <- predict(
  linear_model,
  newdata = test_model
)

linear_metrics <- evaluate_model(
  test_model$cnt,
  linear_predictions
)

linear_metrics$Model <- "Linear Regression"


cat("\n==============================\n")
cat("LINEAR REGRESSION\n")
cat("==============================\n")

print(linear_metrics)


# ============================================================
# 8. RANDOM FOREST
# ============================================================

set.seed(42)

rf_model <- randomForest(
  cnt ~ .,
  data = train_model,
  ntree = 300,
  importance = TRUE
)

rf_predictions <- predict(
  rf_model,
  newdata = test_model
)

rf_metrics <- evaluate_model(
  test_model$cnt,
  rf_predictions
)

rf_metrics$Model <- "Random Forest"


cat("\n==============================\n")
cat("RANDOM FOREST\n")
cat("==============================\n")

print(rf_metrics)


# ============================================================
# 9. GRADIENT BOOSTING
# ============================================================

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


cat("\n==============================\n")
cat("GRADIENT BOOSTING\n")
cat("==============================\n")

print(gbm_metrics)


# ============================================================
# 10. XGBOOST DATA PREPARATION
# ============================================================

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


cat("\nXGBoost training dimensions:\n")
cat("Rows:", nrow(x_train), "\n")
cat("Columns:", ncol(x_train), "\n")


# ============================================================
# 11. XGBOOST
# ============================================================

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


cat("\n==============================\n")
cat("XGBOOST\n")
cat("==============================\n")

print(xgb_metrics)


# ============================================================
# 12. TIME-BASED MODEL LEADERBOARD
# ============================================================

time_results <- bind_rows(
  linear_metrics,
  rf_metrics,
  gbm_metrics,
  xgb_metrics
) %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  ) %>%
  arrange(RMSE)


cat("\n========================================\n")
cat("TIME-BASED MODEL LEADERBOARD\n")
cat("========================================\n")

print(time_results)


# ============================================================
# 13. COMPARE RANDOM SPLIT VS TIME SPLIT
# ============================================================

random_split_results <- data.frame(
  
  Model = c(
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
  ),
  
  Random_Split_RMSE = c(
    99.151,
    47.634,
    43.624,
    47.446
  ),
  
  Random_Split_R2 = c(
    0.6982,
    0.9304,
    0.9416,
    0.9309
  )
)


time_comparison <- time_results %>%
  left_join(
    random_split_results,
    by = "Model"
  ) %>%
  select(
    Model,
    Random_Split_RMSE,
    RMSE,
    Random_Split_R2,
    R_squared
  )


cat("\n========================================\n")
cat("RANDOM SPLIT VS TIME-BASED VALIDATION\n")
cat("========================================\n")

print(time_comparison)


# ============================================================
# 14. ACTUAL VS PREDICTED - BEST MODEL
# ============================================================

best_model <- time_results$Model[1]

if (best_model == "Gradient Boosting") {
  
  best_predictions <- gbm_predictions
  
} else if (best_model == "XGBoost") {
  
  best_predictions <- xgb_predictions
  
} else if (best_model == "Random Forest") {
  
  best_predictions <- rf_predictions
  
} else {
  
  best_predictions <- linear_predictions
  
}


actual_vs_predicted <- data.frame(
  Date = test$date,
  Actual = test$cnt,
  Predicted = best_predictions
)


ggplot(
  actual_vs_predicted,
  aes(
    x = Date
  )
) +
  geom_line(
    aes(y = Actual),
    alpha = 0.6
  ) +
  geom_line(
    aes(y = Predicted),
    alpha = 0.8
  ) +
  labs(
    title = paste(
      "Actual vs Predicted Demand:",
      best_model
    ),
    x = "Date",
    y = "Bike Rental Demand"
  ) +
  theme_minimal()


# ============================================================
# 15. SAVE RESULTS
# ============================================================

write.csv(
  time_results,
  "outputs/time_based_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  time_comparison,
  "outputs/random_vs_time_validation.csv",
  row.names = FALSE
)

write.csv(
  actual_vs_predicted,
  "outputs/time_based_predictions.csv",
  row.names = FALSE
)


cat("\nResults saved to outputs/\n")


# ============================================================
# END
# ============================================================