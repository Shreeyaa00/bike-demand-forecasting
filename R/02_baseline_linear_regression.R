# ============================================================
# Bike Rental Demand Forecasting
# 02 - Baseline Linear Regression
# ============================================================

# Packages ---------------------------------------------------

library(tidyverse)
library(caret)


# ------------------------------------------------------------
# 1. Load Clean Data
# ------------------------------------------------------------

df <- read.csv("data/bike_clean.csv")

# Convert categorical variables back to factors
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


# ------------------------------------------------------------
# 2. Create Baseline Train/Test Split
# ------------------------------------------------------------

set.seed(42)

trainIndex <- createDataPartition(
  df$cnt,
  p = 0.80,
  list = FALSE
)

train <- df[trainIndex, ]
test <- df[-trainIndex, ]


# Check split
cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")


# ------------------------------------------------------------
# 3. Remove Date from Baseline Model
# ------------------------------------------------------------

# Date is retained in the dataset for future time-based
# validation, but is not directly used in the baseline model.

train_model <- train %>%
  select(-date)

test_model <- test %>%
  select(-date)


# ------------------------------------------------------------
# 4. Fit Linear Regression
# ------------------------------------------------------------

linear_model <- lm(
  cnt ~ .,
  data = train_model
)

# Model summary
summary(linear_model)


# ------------------------------------------------------------
# 5. Generate Predictions
# ------------------------------------------------------------

predictions <- predict(
  linear_model,
  newdata = test_model
)


# ------------------------------------------------------------
# 6. Evaluate Model
# ------------------------------------------------------------

actual <- test_model$cnt

RMSE <- sqrt(mean((actual - predictions)^2))

MAE <- mean(abs(actual - predictions))

R_squared <- 1 -
  sum((actual - predictions)^2) /
  sum((actual - mean(actual))^2)


# Display metrics
cat("\n==============================\n")
cat("BASELINE LINEAR REGRESSION\n")
cat("==============================\n")

cat("RMSE:", round(RMSE, 3), "\n")
cat("MAE:", round(MAE, 3), "\n")
cat("R-squared:", round(R_squared, 4), "\n")


# ------------------------------------------------------------
# 7. Actual vs Predicted
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8. Residual Plot
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 9. Save Model Results
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# END
# ------------------------------------------------------------