library(tidyverse)
library(caret)

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

linear_model <- lm(
  cnt ~ .,
  data = train_model
)
summary(linear_model)

predictions <- predict(
  linear_model,
  newdata = test_model
)

actual <- test_model$cnt
RMSE <- sqrt(mean((actual - predictions)^2))
MAE <- mean(abs(actual - predictions))
R_squared <- 1 -
  sum((actual - predictions)^2) /
  sum((actual - mean(actual))^2)

cat("BASELINE LINEAR REGRESSION\n")
cat("RMSE:", round(RMSE, 3), "\n")
cat("MAE:", round(MAE, 3), "\n")
cat("R-squared:", round(R_squared, 4), "\n")

results <- data.frame(
  Actual = actual,
  Predicted = predictions,
  Residual = actual - predictions
)

ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.35) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Linear Regression: Actual vs Predicted",
    x = "Actual Rentals",
    y = "Predicted Rentals"
  ) +
  theme_minimal()

ggplot(results, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.35) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Linear Regression: Residual Analysis",
    x = "Predicted Rentals",
    y = "Residual"
  ) +
  theme_minimal()

model_metrics <- data.frame(
  Model = "Linear Regression",
  RMSE = RMSE,
  MAE = MAE,
  R_squared = R_squared
)

write.csv(
  model_metrics,
  "outputs/linear_regression_metrics.csv",
  row.names = FALSE
)

write.csv(
  results,
  "outputs/linear_regression_predictions.csv",
  row.names = FALSE
)
