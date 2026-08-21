library(tidyverse)
library(caret)
library(gbm)

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

df <- df %>%
  mutate(
    time_index = row_number()
  )

df <- df %>%
  mutate(
    
    lag_1 = lag(cnt, 1),
    
    lag_24 = lag(cnt, 24),
    
    lag_168 = lag(cnt, 168),
    
    lag_3 = lag(cnt, 3),
    
    lag_6 = lag(cnt, 6)
    
  )


df <- df %>%
  mutate(
    
    rolling_mean_3 =
      zoo::rollmean(
        lag(cnt, 1),
        k = 3,
        fill = NA,
        align = "right"
      ),
    
    rolling_mean_24 =
      zoo::rollmean(
        lag(cnt, 1),
        k = 24,
        fill = NA,
        align = "right"
      ),
    
    rolling_mean_168 =
      zoo::rollmean(
        lag(cnt, 1),
        k = 168,
        fill = NA,
        align = "right"
      )
    
  )

cat("NEW FEATURE ENGINEERING\n")
print(
  df %>%
    select(
      date,
      hr,
      cnt,
      lag_1,
      lag_24,
      lag_168,
      lag_3,
      lag_6,
      rolling_mean_3,
      rolling_mean_24,
      rolling_mean_168
    ) %>%
    head(15)
)

df_model <- df %>%
  drop_na(
    lag_1,
    lag_24,
    lag_168,
    rolling_mean_3,
    rolling_mean_24,
    rolling_mean_168
  )


cat("\nOriginal observations:", nrow(df), "\n")
cat("Model observations:", nrow(df_model), "\n")
cat(
  "Observations removed:",
  nrow(df) - nrow(df_model),
  "\n"
)

unique_dates <- sort(
  unique(df_model$date)
)

split_date_index <- floor(
  0.80 * length(unique_dates)
)

split_date <- unique_dates[
  split_date_index
]


train <- df_model %>%
  filter(date < split_date)

test <- df_model %>%
  filter(date >= split_date)

cat("DATE-BASED TRAIN / TEST SPLIT\n")
cat("\nSplit date:", as.character(split_date), "\n")
cat("\nTraining observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")
cat("\nTraining period:\n")
print(range(train$date))
cat("\nTesting period:\n")
print(range(test$date))

date_overlap <- intersect(
  unique(train$date),
  unique(test$date)
)

cat("\nDate overlap between train and test:", length(date_overlap), "\n")

if (length(date_overlap) > 0) {
  stop("ERROR: Train/test date leakage detected.")
} else {
  cat("✓ No date overlap detected.\n")
}

train_model <- train %>%
  select(
    -date,
    -time_index
  )

test_model <- test %>%
  select(
    -date,
    -time_index
  )

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

baseline_train <- train_model %>%
  select(
    -lag_1,
    -lag_24,
    -lag_168,
    -lag_3,
    -lag_6,
    -rolling_mean_3,
    -rolling_mean_24,
    -rolling_mean_168
  )

baseline_test <- test_model %>%
  select(
    -lag_1,
    -lag_24,
    -lag_168,
    -lag_3,
    -lag_6,
    -rolling_mean_3,
    -rolling_mean_24,
    -rolling_mean_168
  )


set.seed(42)

baseline_gbm <- gbm(
  cnt ~ .,
  data = baseline_train,
  distribution = "gaussian",
  n.trees = 500,
  interaction.depth = 5,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)


baseline_predictions <- predict(
  baseline_gbm,
  newdata = baseline_test,
  n.trees = 500
)


baseline_metrics <- evaluate_model(
  baseline_test$cnt,
  baseline_predictions
)

baseline_metrics$Model <-
  "Gradient Boosting - No Lag Features"

set.seed(42)

lag_gbm <- gbm(
  cnt ~ .,
  data = train_model,
  distribution = "gaussian",
  n.trees = 500,
  interaction.depth = 5,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)


lag_predictions <- predict(
  lag_gbm,
  newdata = test_model,
  n.trees = 500
)


lag_metrics <- evaluate_model(
  test_model$cnt,
  lag_predictions
)

lag_metrics$Model <-
  "Gradient Boosting - Lag Features"

lag_comparison <- bind_rows(
  baseline_metrics,
  lag_metrics
) %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  ) %>%
  arrange(RMSE)

cat("LAG FEATURE MODEL COMPARISON\n")
print(lag_comparison)

rmse_improvement <- (
  baseline_metrics$RMSE -
    lag_metrics$RMSE
) / baseline_metrics$RMSE * 100


mae_improvement <- (
  baseline_metrics$MAE -
    lag_metrics$MAE
) / baseline_metrics$MAE * 100


r2_improvement <- (
  lag_metrics$R_squared -
    baseline_metrics$R_squared
)
cat("FEATURE ENGINEERING IMPACT\n")
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

importance <- summary(
  lag_gbm,
  plotit = FALSE
)
cat("GRADIENT BOOSTING FEATURE IMPORTANCE\n")
print(
  head(importance, 20)
)

predictions_df <- data.frame(
  Date = test$date,
  Actual = test$cnt,
  Predicted = lag_predictions
)


prediction_plot <- ggplot(
  predictions_df,
  aes(x = Date)
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
    title = "Actual vs Predicted Bike Rental Demand",
    subtitle = "Gradient Boosting with Historical Demand Features",
    x = "Date",
    y = "Bike Rental Demand"
  ) +
  
  theme_minimal()


print(prediction_plot)

write.csv(
  lag_comparison,
  "outputs/lag_feature_comparison.csv",
  row.names = FALSE
)

write.csv(
  importance,
  "outputs/lag_feature_importance.csv",
  row.names = FALSE
)

write.csv(
  predictions_df,
  "outputs/lag_feature_predictions.csv",
  row.names = FALSE
)


cat("\nResults saved to outputs/\n")
