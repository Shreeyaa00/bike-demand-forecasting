library(tidyverse)

if (!dir.exists("outputs")) {
  dir.create("outputs", recursive = TRUE)
}

predictions <- read.csv(
  "outputs/final_predictions.csv",
  stringsAsFactors = FALSE
)
predictions$datetime <- as.POSIXct(
  predictions$datetime,
  format = "%Y-%m-%d %H:%M:%S",
  tz = "UTC"
)

missing_datetime <- is.na(predictions$datetime)
if (any(missing_datetime)) {
  
  predictions$datetime[missing_datetime] <- as.POSIXct(
    as.character(
      read.csv(
        "outputs/final_predictions.csv",
        stringsAsFactors = FALSE
      )$datetime[missing_datetime]
    ),
    format = "%Y-%m-%d",
    tz = "UTC"
  )
}

predictions <- predictions %>%
  filter(
    !is.na(datetime),
    !is.na(Actual),
    !is.na(Predicted)
  ) %>%
  arrange(datetime)

predictions <- predictions %>%
  mutate(
    Error = Actual - Predicted,
    Absolute_Error = abs(Error)
  )
cat("VISUALIZATION DATA CHECK\n")
cat("Observations:", nrow(predictions), "\n")
cat(
  "Date range:",
  format(min(predictions$datetime), "%Y-%m-%d"),
  "to",
  format(max(predictions$datetime), "%Y-%m-%d"),
  "\n"
)

p1 <- ggplot(
  predictions,
  aes(x = datetime)
) +
  geom_line(
    aes(y = Actual, linetype = "Actual"),
    linewidth = 0.6
  ) +
  geom_line(
    aes(y = Predicted, linetype = "Predicted"),
    linewidth = 0.6
  ) +
  labs(
    title = "Actual vs Predicted Bike Rental Demand",
    subtitle = "Final Tuned Calendar-Aware Gradient Boosting Model",
    x = "Date",
    y = "Rental Count",
    linetype = "Series"
  ) +
  theme_minimal()

ggsave(
  "outputs/01_actual_vs_predicted.png",
  p1,
  width = 12,
  height = 6,
  dpi = 300
)

p2 <- ggplot(
  predictions,
  aes(x = datetime, y = Error)
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  geom_line(
    linewidth = 0.5
  ) +
  labs(
    title = "Prediction Error Over Time",
    subtitle = "Positive = Underprediction | Negative = Overprediction",
    x = "Date",
    y = "Prediction Error"
  ) +
  theme_minimal()

ggsave(
  "outputs/02_prediction_error_over_time.png",
  p2,
  width = 12,
  height = 6,
  dpi = 300
)

p3 <- ggplot(
  predictions,
  aes(x = Error)
) +
  geom_histogram(
    bins = 50
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Distribution of Prediction Errors",
    x = "Prediction Error",
    y = "Frequency"
  ) +
  theme_minimal()

ggsave(
  "outputs/03_error_distribution.png",
  p3,
  width = 10,
  height = 6,
  dpi = 300
)

p4 <- ggplot(
  predictions,
  aes(x = Actual, y = Predicted)
) +
  geom_point(
    alpha = 0.35
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted Demand",
    subtitle = "Points closer to the diagonal indicate better predictions",
    x = "Actual Rental Count",
    y = "Predicted Rental Count"
  ) +
  theme_minimal()

ggsave(
  "outputs/04_actual_vs_predicted_scatter.png",
  p4,
  width = 8,
  height = 8,
  dpi = 300
)

p5 <- ggplot(
  predictions,
  aes(x = datetime, y = Absolute_Error)
) +
  geom_line(
    linewidth = 0.5
  ) +
  labs(
    title = "Absolute Prediction Error Over Time",
    x = "Date",
    y = "Absolute Error"
  ) +
  theme_minimal()

ggsave(
  "outputs/05_absolute_error_over_time.png",
  p5,
  width = 12,
  height = 6,
  dpi = 300
)

top_errors <- predictions %>%
  arrange(desc(Absolute_Error)) %>%
  slice_head(n = 20) %>%
  mutate(
    datetime_label = format(datetime, "%Y-%m-%d %H:%M")
  )

p6 <- ggplot(
  top_errors,
  aes(
    x = reorder(datetime_label, Absolute_Error),
    y = Absolute_Error
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 20 Highest Prediction Errors",
    x = "Date and Hour",
    y = "Absolute Error"
  ) +
  theme_minimal()

ggsave(
  "outputs/06_top_20_prediction_errors.png",
  p6,
  width = 10,
  height = 8,
  dpi = 300
)
cat("VISUALIZATION GENERATION COMPLETE\n")
created_files <- list.files(
  "outputs",
  pattern = "^0[1-6]_.*\\.png$",
  full.names = FALSE
)

print(created_files)

cat("\nTotal visualizations created:", length(created_files), "\n")

cat("\nFiles saved to outputs/\n")