library(readr)
library(dplyr)
library(lubridate)

predictions <- read_csv(
  "outputs/final_predictions.csv",
  show_col_types = FALSE
)

predictions$datetime <- as.POSIXct(
  predictions$datetime,
  format = "%Y-%m-%d %H:%M:%S %Z",
  tz = "UTC"
)

cat("Prediction observations:", nrow(predictions), "\n")

cat(
  "Missing prediction datetimes:",
  sum(is.na(predictions$datetime)),
  "\n"
)

cat(
  "Duplicate prediction datetimes:",
  sum(duplicated(predictions$datetime)),
  "\n"
)

bike_data <- read_csv(
  "data/bike_sharing.csv",
  show_col_types = FALSE
)

bike_data <- bike_data %>%
  mutate(
    datetime = as.POSIXct(
      paste(
        dteday,
        sprintf("%02d:00:00", hr)
      ),
      format = "%Y-%m-%d %H:%M:%S",
      tz = "UTC"
    )
  )

cat("Original observations:", nrow(bike_data), "\n")

feature_data <- bike_data %>%
  select(
    datetime,
    hr,
    mnth,
    season,
    holiday,
    workingday,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed
  )

powerbi_data <- predictions %>%
  inner_join(
    feature_data,
    by = "datetime"
  ) %>%
  arrange(datetime)

cat("MATCHING CHECK\n")
cat(
  "Prediction rows:",
  nrow(predictions),
  "\n"
)

cat(
  "Matched rows:",
  nrow(powerbi_data),
  "\n"
)

cat(
  "Unmatched predictions:",
  nrow(predictions) - nrow(powerbi_data),
  "\n"
)
if (nrow(powerbi_data) != nrow(predictions)) {
  
  unmatched <- predictions %>%
    anti_join(
      feature_data,
      by = "datetime"
    )
  
  cat("\nWARNING: Some predictions could not be matched.\n")
  print(unmatched)
  
  stop(
    "Prediction matching failed. Do not import into Power BI yet."
  )
}
powerbi_data <- powerbi_data %>%
  mutate(
    error = Actual - Predicted,
    
    absolute_error = abs(error),
    
    percent_error = if_else(
      Actual != 0,
      abs(error) / Actual * 100,
      NA_real_
    ),
    
    peak_demand = Actual >= 615.9
  )

powerbi_data <- powerbi_data %>%
  mutate(
    hour = hr,
    
    month = mnth,
    
    Working_Day_Label = case_when(
      workingday == 1 ~ "Working Day",
      workingday == 0 ~ "Non-Working Day",
      TRUE ~ "Unknown"
    ),
    
    Weather_Label = case_when(
      weathersit == 1 ~ "Clear / Partly Cloudy",
      weathersit == 2 ~ "Mist / Cloudy",
      weathersit == 3 ~ "Light Rain / Snow",
      weathersit == 4 ~ "Heavy Rain / Snow",
      TRUE ~ "Unknown"
    ),
    
    Peak_Demand_Label = case_when(
      peak_demand ~ "Peak Demand",
      TRUE ~ "Normal Demand"
    )
  )

powerbi_data <- powerbi_data %>%
  select(
    datetime,
    Actual,
    Predicted,
    hr,
    mnth,
    season,
    holiday,
    workingday,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed,
    error,
    absolute_error,
    percent_error,
    peak_demand,
    hour,
    month,
    Working_Day_Label,
    Weather_Label,
    Peak_Demand_Label
  )

cat("POWER BI DATASET SUMMARY\n")
cat(
  "Observations:",
  nrow(powerbi_data),
  "\n"
)

cat(
  "Columns:",
  ncol(powerbi_data),
  "\n"
)

cat(
  "Date range:",
  as.character(min(powerbi_data$datetime)),
  "to",
  as.character(max(powerbi_data$datetime)),
  "\n"
)

cat(
  "Missing values:",
  sum(is.na(powerbi_data)),
  "\n"
)

cat(
  "Duplicate datetimes:",
  sum(duplicated(powerbi_data$datetime)),
  "\n"
)

cat("\nColumns:\n")
print(names(powerbi_data))

if (
  nrow(powerbi_data) != 3302 ||
  sum(is.na(powerbi_data)) != 0 ||
  sum(duplicated(powerbi_data$datetime)) != 0
) {
  
  stop(
    "FINAL POWER BI DATA VALIDATION FAILED."
  )
}
write_csv(
  powerbi_data,
  "outputs/powerbi_demand_data.csv"
)
