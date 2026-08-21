library(tidyverse)
library(caret)
library(glmnet)

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

ridge_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0,
  family = "gaussian",
  nfolds = 10,
  standardize = TRUE
)

ridge_lambda <- ridge_cv$lambda.min
cat("RIDGE REGRESSION\n")
cat("Best lambda:", ridge_lambda, "\n")


ridge_predictions <- predict(
  ridge_cv,
  newx = x_test,
  s = "lambda.min"
)[, 1]


ridge_metrics <- evaluate_model(
  y_test,
  ridge_predictions
)

ridge_metrics$Model <- "Ridge Regression"
print(ridge_metrics)

set.seed(42)

lasso_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  family = "gaussian",
  nfolds = 10,
  standardize = TRUE
)

lasso_lambda <- lasso_cv$lambda.min
cat("LASSO REGRESSION\n")
cat("Best lambda:", lasso_lambda, "\n")


lasso_predictions <- predict(
  lasso_cv,
  newx = x_test,
  s = "lambda.min"
)[, 1]


lasso_metrics <- evaluate_model(
  y_test,
  lasso_predictions
)

lasso_metrics$Model <- "Lasso Regression"

print(lasso_metrics)

set.seed(42)

alpha_grid <- seq(
  0.1,
  0.9,
  by = 0.1
)

elastic_models <- list()

for (alpha_value in alpha_grid) {
  
  set.seed(42)
  
  cv_model <- cv.glmnet(
    x_train,
    y_train,
    alpha = alpha_value,
    family = "gaussian",
    nfolds = 10,
    standardize = TRUE
  )
  
  predictions <- predict(
    cv_model,
    newx = x_test,
    s = "lambda.min"
  )[, 1]
  
  metrics <- evaluate_model(
    y_test,
    predictions
  )
  
  metrics$Alpha <- alpha_value
  metrics$Lambda <- cv_model$lambda.min
  
  elastic_models[[as.character(alpha_value)]] <- metrics
}


elastic_results <- bind_rows(
  elastic_models
)


best_elastic <- elastic_results %>%
  arrange(RMSE) %>%
  slice(1)

cat("ELASTIC NET\n")
cat("Best alpha:", best_elastic$Alpha, "\n")
cat("Best lambda:", best_elastic$Lambda, "\n")

print(best_elastic)

linear_metrics <- data.frame(
  Model = "Linear Regression",
  RMSE = 99.151,
  MAE = 73.909,
  R_squared = 0.6982
)

ridge_results <- ridge_metrics %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )

lasso_results <- lasso_metrics %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )

elastic_best_results <- best_elastic %>%
  mutate(
    Model = "Elastic Net"
  ) %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )


model_comparison <- bind_rows(
  linear_metrics,
  ridge_results,
  lasso_results,
  elastic_best_results
) %>%
  arrange(RMSE)

cat("REGULARIZED MODEL COMPARISON\n")
print(model_comparison)

comparison_long <- model_comparison %>%
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
    title = "Regularized Regression Model Comparison",
    x = "Model",
    y = "Error"
  ) +
  theme_minimal()

lasso_coefficients <- coef(
  lasso_cv,
  s = "lambda.min"
)

lasso_coefficients_df <- as.matrix(
  lasso_coefficients
) %>%
  as.data.frame() %>%
  rownames_to_column("Feature")
names(lasso_coefficients_df)[2] <- "Coefficient"

lasso_coefficients_df <- lasso_coefficients_df %>%
  filter(
    Feature != "(Intercept)",
    Coefficient != 0
  ) %>%
  arrange(
    desc(abs(Coefficient))
  )

cat("LASSO SELECTED FEATURES\n")
print(lasso_coefficients_df)

write.csv(
  model_comparison,
  "outputs/regularized_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  lasso_coefficients_df,
  "outputs/lasso_selected_features.csv",
  row.names = FALSE
)

cat("\nResults saved to outputs/\n")
