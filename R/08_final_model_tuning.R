library(tidyverse)
library(gbm)
library(zoo)

df <- read.csv("data/bike_clean.csv")

df <- df %>%
  mutate(
    date = as.Date(date),
    hour = as.integer(as.character(hr))
  ) %>%
  arrange(date, hour)

df <- df %>%
  mutate(
    datetime = as.POSIXct(
      paste(date, sprintf("%02d:00:00", hour)),
      format = "%Y-%m-%d %H:%M:%S",
      tz = "UTC"
    )
  )

demand_lookup <- df %>%
  select(datetime, cnt)

df <- df %>%
  mutate(
    previous_hour = datetime - 60 * 60
  ) %>%
  left_join(
    demand_lookup %>%
      rename(
        previous_hour = datetime,
        lag_1h = cnt
      ),
    by = "previous_hour"
  ) %>%
  select(-previous_hour)

df <- df %>%
  mutate(
    previous_day = datetime - 24 * 60 * 60
  ) %>%
  left_join(
    demand_lookup %>%
      rename(
        previous_day = datetime,
        lag_24h = cnt
      ),
    by = "previous_day"
  ) %>%
  select(-previous_day)

df <- df %>%
  mutate(
    previous_week = datetime - 7 * 24 * 60 * 60
  ) %>%
  left_join(
    demand_lookup %>%
      rename(
        previous_week = datetime,
        lag_168h = cnt
      ),
    by = "previous_week"
  ) %>%
  select(-previous_week)

df <- df %>%
  arrange(datetime) %>%
  mutate(
    rolling_24h = zoo::rollmean(
      lag(cnt, 1),
      k = 24,
      fill = NA,
      align = "right"
    ),
    
    rolling_7d = zoo::rollmean(
      lag(cnt, 1),
      k = 168,
      fill = NA,
      align = "right"
    )
  )

df_model <- df %>%
  drop_na(
    lag_1h,
    lag_24h,
    lag_168h,
    rolling_24h,
    rolling_7d
  )
cat("FEATURE ENGINEERING SUMMARY\n")
cat("Original observations:", nrow(df), "\n")
cat("Model observations:", nrow(df_model), "\n")
cat(
  "Observations removed:",
  nrow(df) - nrow(df_model),
  "\n"
)

unique_dates <- sort(unique(df_model$date))

split_index <- floor(
  0.80 * length(unique_dates)
)

split_date <- unique_dates[split_index]

train <- df_model %>%
  filter(date < split_date)

test <- df_model %>%
  filter(date >= split_date)
cat("FINAL DATE-BASED SPLIT\n")
cat("Split date:", as.character(split_date), "\n")
cat(
  "Training observations:",
  nrow(train),
  "\n"
)

cat(
  "Testing observations:",
  nrow(test),
  "\n"
)

cat("\nTraining period:\n")
print(range(train$date))

cat("\nTesting period:\n")
print(range(test$date))

overlap <- intersect(
  unique(train$date),
  unique(test$date)
)

cat(
  "\nDate overlap:",
  length(overlap),
  "\n"
)

if (length(overlap) > 0) {
  stop("ERROR: Date overlap detected.")
}

cat("✓ No date overlap detected.\n")

train_model <- train %>%
  select(
    -cnt,
    -date,
    -datetime,
    -hour
  )

test_model <- test %>%
  select(
    -cnt,
    -date,
    -datetime,
    -hour
  )
train_model <- train_model %>%
  mutate(
    across(
      where(is.character),
      as.factor
    )
  )

test_model <- test_model %>%
  mutate(
    across(
      where(is.character),
      as.factor
    )
  )

dummy_model <- caret::dummyVars(
  ~ .,
  data = train_model,
  fullRank = TRUE
)

x_train <- predict(
  dummy_model,
  newdata = train_model
)

x_test <- predict(
  dummy_model,
  newdata = test_model
)

x_train <- as.data.frame(x_train)
x_test <- as.data.frame(x_test)

y_train <- train$cnt
y_test <- test$cnt

train_gbm <- x_train %>%
  mutate(
    target = y_train
  )

test_gbm <- x_test

gbm_formula <- as.formula(
  paste(
    "target ~",
    paste(
      colnames(x_train),
      collapse = " + "
    )
  )
)

evaluate_model <- function(
    actual,
    predicted,
    model_name) {
  
  rmse <- sqrt(
    mean(
      (actual - predicted)^2
    )
  )
  
  mae <- mean(
    abs(actual - predicted)
  )
  
  r_squared <- 1 -
    sum(
      (actual - predicted)^2
    ) /
    sum(
      (actual - mean(actual))^2
    )
  
  data.frame(
    Model = model_name,
    RMSE = rmse,
    MAE = mae,
    R_squared = r_squared
  )
}

set.seed(42)

baseline_gbm <- gbm(
  formula = gbm_formula,
  data = train_gbm,
  distribution = "gaussian",
  n.trees = 500,
  interaction.depth = 5,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)

baseline_predictions <- predict(
  baseline_gbm,
  newdata = test_gbm,
  n.trees = 500
)

baseline_metrics <- evaluate_model(
  y_test,
  baseline_predictions,
  "Calendar-Aware GBM"
)
cat("CALENDAR-AWARE BASELINE\n")
print(baseline_metrics)

tuning_grid <- expand.grid(
  n.trees = c(300, 500, 700),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.03, 0.05),
  n.minobsinnode = c(10, 20)
)

tuning_results <- data.frame()
cat("HYPERPARAMETER TUNING\n")
set.seed(42)

for (i in seq_len(nrow(tuning_grid))) {
  
  params <- tuning_grid[i, ]
  
  cat(
    "\nTesting model",
    i,
    "of",
    nrow(tuning_grid),
    "\n"
  )
  
  model <- gbm(
    formula = gbm_formula,
    data = train_gbm,
    distribution = "gaussian",
    n.trees = params$n.trees,
    interaction.depth =
      params$interaction.depth,
    shrinkage =
      params$shrinkage,
    n.minobsinnode =
      params$n.minobsinnode,
    verbose = FALSE
  )
  
  predictions <- predict(
    model,
    newdata = test_gbm,
    n.trees = params$n.trees
  )
  
  metrics <- evaluate_model(
    y_test,
    predictions,
    "GBM"
  )
  
  tuning_results <- bind_rows(
    tuning_results,
    data.frame(
      n.trees =
        params$n.trees,
      interaction.depth =
        params$interaction.depth,
      shrinkage =
        params$shrinkage,
      n.minobsinnode =
        params$n.minobsinnode,
      RMSE =
        metrics$RMSE,
      MAE =
        metrics$MAE,
      R_squared =
        metrics$R_squared
    )
  )
}

tuning_results <- tuning_results %>%
  arrange(RMSE)
cat("TOP 10 TUNING RESULTS\n")

print(
  head(
    tuning_results,
    10
  )
)

best_params <- tuning_results %>%
  slice(1)
cat("BEST HYPERPARAMETERS\n")
print(best_params)

set.seed(42)

final_model <- gbm(
  formula = gbm_formula,
  data = train_gbm,
  distribution = "gaussian",
  n.trees =
    best_params$n.trees,
  interaction.depth =
    best_params$interaction.depth,
  shrinkage =
    best_params$shrinkage,
  n.minobsinnode =
    best_params$n.minobsinnode,
  verbose = FALSE
)

final_predictions <- predict(
  final_model,
  newdata = test_gbm,
  n.trees =
    best_params$n.trees
)

final_metrics <- evaluate_model(
  y_test,
  final_predictions,
  "Tuned Calendar-Aware GBM"
)
cat("FINAL TUNED MODEL\n")
print(final_metrics)

final_comparison <- bind_rows(
  baseline_metrics,
  final_metrics
) %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  ) %>%
  arrange(RMSE)
cat("BASELINE VS TUNED MODEL\n")
print(final_comparison)

rmse_improvement <- (
  baseline_metrics$RMSE -
    final_metrics$RMSE
) /
  baseline_metrics$RMSE *
  100

mae_improvement <- (
  baseline_metrics$MAE -
    final_metrics$MAE
) /
  baseline_metrics$MAE *
  100

r2_improvement <-
  final_metrics$R_squared -
  baseline_metrics$R_squared

cat("TUNING IMPACT\n")
cat(
  "RMSE improvement:",
  round(rmse_improvement, 2),
  "%\n"
)

cat(
  "MAE improvement:",
  round(mae_improvement, 2),
  "%\n"
)

cat(
  "R-squared improvement:",
  round(r2_improvement, 4),
  "\n"
)

final_importance <- summary(
  final_model,
  plotit = FALSE
)

cat("FINAL FEATURE IMPORTANCE\n")
print(
  head(
    final_importance,
    20
  )
)

prediction_df <- data.frame(
  datetime = test$datetime,
  Actual = y_test,
  Predicted = final_predictions
)

prediction_plot <- ggplot(
  prediction_df,
  aes(x = datetime)
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
    title =
      "Actual vs Predicted Bike Rental Demand",
    subtitle =
      "Tuned Calendar-Aware Gradient Boosting",
    x = "Date",
    y = "Bike Rental Demand"
  ) +
  theme_minimal()

print(prediction_plot)

write.csv(
  tuning_results,
  "outputs/gbm_tuning_results.csv",
  row.names = FALSE
)

write.csv(
  final_comparison,
  "outputs/final_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  final_importance,
  "outputs/final_feature_importance.csv",
  row.names = FALSE
)

write.csv(
  prediction_df,
  "outputs/final_predictions.csv",
  row.names = FALSE
)
cat("FINAL MODELING PHASE COMPLETE\n")
cat("\nResults saved to outputs/\n")