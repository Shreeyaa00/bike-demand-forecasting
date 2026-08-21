library(tidyverse)

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
  original_dates <- read.csv(
    "outputs/final_predictions.csv",
    stringsAsFactors = FALSE
  )$datetime
  
  predictions$datetime[missing_datetime] <- as.POSIXct(
    original_dates[missing_datetime],
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
  mutate(
    hour = as.integer(format(datetime, "%H")),
    month = as.integer(format(datetime, "%m")),
    weekday = weekdays(datetime),
    
    Error = Actual - Predicted,
    Absolute_Error = abs(Error),
    
    Peak_Demand = Actual >= quantile(
      Actual,
      0.90,
      na.rm = TRUE
    )
  )

overall_metrics <- predictions %>%
  summarise(
    Total_Observations = n(),
    Average_Actual_Demand = mean(Actual),
    Average_Predicted_Demand = mean(Predicted),
    Maximum_Actual_Demand = max(Actual),
    Maximum_Predicted_Demand = max(Predicted),
    Average_Absolute_Error = mean(Absolute_Error),
    RMSE = sqrt(mean((Actual - Predicted)^2)),
    R_Squared = cor(Actual, Predicted)^2
  )
cat("OVERALL BUSINESS METRICS\n")
print(overall_metrics)

hour_summary <- predictions %>%
  group_by(hour) %>%
  summarise(
    Average_Demand = mean(Actual),
    Maximum_Demand = max(Actual),
    Average_Predicted = mean(Predicted),
    Average_Error = mean(Error),
    MAE = mean(Absolute_Error),
    .groups = "drop"
  ) %>%
  arrange(desc(Average_Demand))
cat("DEMAND BY HOUR\n")
print(hour_summary)

top_hours <- hour_summary %>%
  arrange(desc(Average_Demand)) %>%
  slice_head(n = 5)
cat("TOP 5 DEMAND HOURS\n")
print(top_hours)

month_summary <- predictions %>%
  group_by(month) %>%
  summarise(
    Average_Demand = mean(Actual),
    Maximum_Demand = max(Actual),
    Average_Predicted = mean(Predicted),
    MAE = mean(Absolute_Error),
    .groups = "drop"
  ) %>%
  arrange(desc(Average_Demand))
cat("DEMAND BY MONTH\n")
print(month_summary)

peak_summary <- predictions %>%
  group_by(Peak_Demand) %>%
  summarise(
    Observations = n(),
    Average_Actual = mean(Actual),
    Average_Predicted = mean(Predicted),
    RMSE = sqrt(mean((Actual - Predicted)^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    .groups = "drop"
  )
cat("PEAK DEMAND ANALYSIS\n")
print(peak_summary)

peak_threshold <- quantile(
  predictions$Actual,
  0.90,
  na.rm = TRUE
)

cat("\nPeak demand threshold:", round(peak_threshold, 2), "\n")

top_demand_periods <- predictions %>%
  arrange(desc(Actual)) %>%
  slice_head(n = 20) %>%
  select(
    datetime,
    Actual,
    Predicted,
    Error,
    Absolute_Error,
    hour,
    month
  )
cat("TOP 20 HIGHEST DEMAND PERIODS\n")
print(top_demand_periods)

top_model_errors <- predictions %>%
  arrange(desc(Absolute_Error)) %>%
  slice_head(n = 20) %>%
  select(
    datetime,
    Actual,
    Predicted,
    Error,
    Absolute_Error,
    hour,
    month
  )
cat("TOP 20 MODEL FAILURES\n")
print(top_model_errors)

hour_variability <- predictions %>%
  group_by(hour) %>%
  summarise(
    Mean_Demand = mean(Actual),
    SD_Demand = sd(Actual),
    Max_Demand = max(Actual),
    .groups = "drop"
  ) %>%
  arrange(desc(SD_Demand))
cat("DEMAND VARIABILITY BY HOUR\n")
print(hour_variability)

write.csv(
  overall_metrics,
  "outputs/business_overall_metrics.csv",
  row.names = FALSE
)

write.csv(
  hour_summary,
  "outputs/business_hour_summary.csv",
  row.names = FALSE
)

write.csv(
  month_summary,
  "outputs/business_month_summary.csv",
  row.names = FALSE
)

write.csv(
  peak_summary,
  "outputs/business_peak_summary.csv",
  row.names = FALSE
)

write.csv(
  top_demand_periods,
  "outputs/business_top_demand_periods.csv",
  row.names = FALSE
)

write.csv(
  top_model_errors,
  "outputs/business_top_model_errors.csv",
  row.names = FALSE
)

write.csv(
  hour_variability,
  "outputs/business_hour_variability.csv",
  row.names = FALSE
)
cat("BUSINESS INSIGHTS ANALYSIS COMPLETE\n")
cat("\nBusiness insight files saved to outputs/\n")