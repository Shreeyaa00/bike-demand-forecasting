library(tidyverse)
library(lubridate)
library(zoo)

predictions <- read.csv(
  "outputs/final_predictions.csv",
  stringsAsFactors = FALSE
)

predictions <- predictions %>%
  mutate(
    datetime = as.character(datetime),
    datetime = parse_date_time(
      datetime,
      orders = c(
        "Y-m-d H:M:S",
        "Y-m-d H:M",
        "Y-m-d"
      )
    )
  )

df <- read.csv(
  "data/bike_clean.csv",
  stringsAsFactors = FALSE
)

df$date <- as.Date(df$date)

df <- df %>%
  mutate(
    hr_numeric = as.integer(as.character(hr)),
    datetime = as.POSIXct(
      paste(
        date,
        sprintf("%02d:00:00", hr_numeric)
      ),
      format = "%Y-%m-%d %H:%M:%S"
    )
  )

df <- df %>%
  arrange(datetime) %>%
  mutate(
    lag_1h = lag(cnt, 1),
    lag_24h = lag(cnt, 24),
    lag_168h = lag(cnt, 168),
    
    rolling_24h = lag(
      zoo::rollmean(
        cnt,
        24,
        fill = NA,
        align = "right"
      ),
      1
    ),
    
    rolling_7d = lag(
      zoo::rollmean(
        cnt,
        168,
        fill = NA,
        align = "right"
      ),
      1
    )
  )

model_df <- df %>%
  filter(
    !is.na(lag_1h),
    !is.na(lag_24h),
    !is.na(lag_168h),
    !is.na(rolling_24h),
    !is.na(rolling_7d)
  )

analysis_df <- predictions %>%
  left_join(
    model_df %>%
      select(
        datetime,
        hr,
        mnth,
        season,
        workingday,
        weathersit,
        temp,
        atemp,
        hum,
        windspeed
      ),
    by = "datetime"
  )

cat("DATETIME MATCHING CHECK\n")
cat(
  "Prediction observations:",
  nrow(predictions),
  "\n"
)

matched <- sum(!is.na(analysis_df$hr))

unmatched <- sum(is.na(analysis_df$hr))

cat(
  "Matched observations:",
  matched,
  "\n"
)

cat(
  "Unmatched observations:",
  unmatched,
  "\n"
)

if (unmatched == 0) {
  cat("✓ All predictions successfully matched.\n")
} else {
  cat("⚠ Some predictions could not be matched.\n")
}
analysis_df <- analysis_df %>%
  mutate(
    Error = Actual - Predicted,
    Absolute_Error = abs(Error),
    Percent_Error = abs(Error) /
      pmax(Actual, 1) * 100
  )

overall_error <- analysis_df %>%
  summarise(
    RMSE = sqrt(mean(Error^2, na.rm = TRUE)),
    MAE = mean(Absolute_Error, na.rm = TRUE),
    Mean_Error = mean(Error, na.rm = TRUE),
    Mean_Percent_Error = mean(
      Percent_Error,
      na.rm = TRUE
    )
  )
cat("OVERALL ERROR ANALYSIS\n")
print(overall_error)

error_by_hour <- analysis_df %>%
  filter(!is.na(hr)) %>%
  group_by(hr) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  )
cat("ERROR BY HOUR\n")
print(error_by_hour, n = 24)

error_by_weather <- analysis_df %>%
  filter(!is.na(weathersit)) %>%
  group_by(weathersit) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  )
cat("ERROR BY WEATHER\n")
print(error_by_weather)

error_by_workingday <- analysis_df %>%
  filter(!is.na(workingday)) %>%
  group_by(workingday) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  )
cat("ERROR BY WORKING DAY\n")
print(error_by_workingday)

error_by_month <- analysis_df %>%
  filter(!is.na(mnth)) %>%
  group_by(mnth) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  ) %>%
  arrange(desc(RMSE))

cat("ERROR BY MONTH\n")
print(error_by_month)

error_by_season <- analysis_df %>%
  filter(!is.na(season)) %>%
  group_by(season) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  )
cat("ERROR BY SEASON\n")
print(error_by_season)

peak_threshold <- quantile(
  analysis_df$Actual,
  0.90,
  na.rm = TRUE
)

analysis_df <- analysis_df %>%
  mutate(
    Peak_Demand = Actual >= peak_threshold
  )

peak_error <- analysis_df %>%
  group_by(Peak_Demand) %>%
  summarise(
    RMSE = sqrt(mean(Error^2)),
    MAE = mean(Absolute_Error),
    Mean_Error = mean(Error),
    Mean_Actual = mean(Actual),
    .groups = "drop"
  )
cat("PEAK DEMAND ERROR\n")
print(peak_error)
cat(
  "\nPeak demand threshold:",
  round(peak_threshold, 2),
  "\n"
)

top_errors <- analysis_df %>%
  arrange(desc(Absolute_Error)) %>%
  select(
    datetime,
    Actual,
    Predicted,
    Error,
    Absolute_Error,
    Percent_Error,
    hr,
    mnth,
    season,
    workingday,
    weathersit,
    temp,
    hum
  ) %>%
  slice_head(n = 20)

cat("TOP 20 HIGHEST-ERROR PREDICTIONS\n")
print(top_errors)

residual_summary <- analysis_df %>%
  summarise(
    Mean_Residual = mean(Error, na.rm = TRUE),
    Median_Residual = median(Error, na.rm = TRUE),
    SD_Residual = sd(Error, na.rm = TRUE),
    Min_Residual = min(Error, na.rm = TRUE),
    Max_Residual = max(Error, na.rm = TRUE)
  )
cat("RESIDUAL SUMMARY\n")
print(residual_summary)

p1 <- ggplot(
  analysis_df,
  aes(x = Actual, y = Predicted)
) +
  geom_point(alpha = 0.35) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted Bike Demand",
    x = "Actual Demand",
    y = "Predicted Demand"
  ) +
  theme_minimal()

ggsave(
  "outputs/actual_vs_predicted.png",
  p1,
  width = 8,
  height = 6
)

p2 <- ggplot(
  analysis_df,
  aes(x = Error)
) +
  geom_histogram(bins = 50) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Residual Distribution",
    x = "Prediction Error",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  "outputs/residual_distribution.png",
  p2,
  width = 8,
  height = 6
)

p3 <- ggplot(
  error_by_hour,
  aes(x = hr, y = RMSE)
) +
  geom_col() +
  labs(
    title = "Prediction Error by Hour",
    x = "Hour of Day",
    y = "RMSE"
  ) +
  theme_minimal()

ggsave(
  "outputs/error_by_hour.png",
  p3,
  width = 8,
  height = 6
)

p4 <- ggplot(
  peak_error,
  aes(x = Peak_Demand, y = RMSE)
) +
  geom_col() +
  labs(
    title = "Peak vs Non-Peak Prediction Error",
    x = "Peak Demand",
    y = "RMSE"
  ) +
  theme_minimal()

ggsave(
  "outputs/peak_demand_error.png",
  p4,
  width = 8,
  height = 6
)
write.csv(
  overall_error,
  "outputs/overall_error.csv",
  row.names = FALSE
)

write.csv(
  error_by_hour,
  "outputs/error_by_hour.csv",
  row.names = FALSE
)

write.csv(
  error_by_weather,
  "outputs/error_by_weather.csv",
  row.names = FALSE
)

write.csv(
  error_by_workingday,
  "outputs/error_by_workingday.csv",
  row.names = FALSE
)

write.csv(
  error_by_month,
  "outputs/error_by_month.csv",
  row.names = FALSE
)

write.csv(
  error_by_season,
  "outputs/error_by_season.csv",
  row.names = FALSE
)

write.csv(
  peak_error,
  "outputs/peak_demand_error.csv",
  row.names = FALSE
)

write.csv(
  top_errors,
  "outputs/top_20_errors.csv",
  row.names = FALSE
)

write.csv(
  analysis_df,
  "outputs/final_error_analysis.csv",
  row.names = FALSE
)
cat("ERROR ANALYSIS COMPLETE\n")
cat("\nResults saved to outputs/\n")