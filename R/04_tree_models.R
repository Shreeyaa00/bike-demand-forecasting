library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

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

# Remove date
train_model <- train %>%
  select(-date)

test_model <- test %>%
  select(-date)
evaluate_model <- function(actual, predicted) {
  
  rmse <- sqrt(mean((actual - predicted)^2))
  
  mae <- mean(abs(actual - predicted))
  
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

tree_model <- rpart(
  cnt ~ .,
  data = train_model,
  method = "anova",
  control = rpart.control(
    cp = 0.001,
    maxdepth = 10
  )
)
rpart.plot(tree_model)

tree_predictions <- predict(
  tree_model,
  newdata = test_model
)

tree_metrics <- evaluate_model(
  test_model$cnt,
  tree_predictions
)

tree_metrics$Model <- "Decision Tree"
cat("DECISION TREE\n")
print(tree_metrics)

set.seed(42)

rf_model <- randomForest(
  cnt ~ .,
  data = train_model,
  ntree = 500,
  importance = TRUE
)

# Predictions
rf_predictions <- predict(
  rf_model,
  newdata = test_model
)

# Metrics
rf_metrics <- evaluate_model(
  test_model$cnt,
  rf_predictions
)

rf_metrics$Model <- "Random Forest"

cat("RANDOM FOREST\n")
print(rf_metrics)

importance_df <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  arrange(desc(`%IncMSE`))
cat("RANDOM FOREST FEATURE IMPORTANCE\n")

print(head(importance_df, 15))

# Plot importance
varImpPlot(rf_model)

previous_models <- data.frame(
  Model = c(
    "Linear Regression",
    "Lasso Regression",
    "Elastic Net",
    "Ridge Regression"
  ),
  RMSE = c(
    99.151,
    99.158,
    99.159,
    100.394
  ),
  MAE = c(
    73.909,
    73.876,
    73.878,
    74.734
  ),
  R_squared = c(
    0.6982,
    0.6982,
    0.6982,
    0.6906
  )
)

tree_results <- tree_metrics %>%
  select(Model, RMSE, MAE, R_squared)

rf_results <- rf_metrics %>%
  select(Model, RMSE, MAE, R_squared)

all_models <- bind_rows(
  previous_models,
  tree_results,
  rf_results
) %>%
  arrange(RMSE)

cat("MODEL LEADERBOARD\n")
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
    title = "Model Comparison",
    x = "Model",
    y = "Error"
  ) +
  theme_minimal()

write.csv(
  all_models,
  "outputs/model_comparison_tree_models.csv",
  row.names = FALSE
)

write.csv(
  importance_df,
  "outputs/random_forest_importance.csv",
  row.names = FALSE
)

cat("\nResults saved to outputs/\n")