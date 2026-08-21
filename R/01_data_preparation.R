library(tidyverse)
library(caret)
library(lubridate)

df_raw <- read.csv("data/bike_sharing.csv")

# Inspect dimensions
dim(df_raw)

# Inspect structure
str(df_raw)

# Preview data
head(df_raw)

df <- df_raw %>%
  mutate(
    date = as.Date(dteday)
  )

# Missing values
colSums(is.na(df))

# Identify incomplete observations
missing_rows <- df %>%
  filter(if_any(everything(), is.na))

print(missing_rows)

# Check duplicate rows
sum(duplicated(df))

# Date range
range(df$date)

# Number of unique dates
n_distinct(df$date)

df <- df %>%
  filter(!is.na(cnt))
colSums(is.na(df))

df <- df %>%
  select(
    -instant,
    -dteday,
    -casual,
    -registered
  )

df <- df %>%
  mutate(
    season = factor(season),
    yr = factor(yr),
    mnth = factor(mnth),
    hr = factor(hr),
    holiday = factor(holiday),
    weekday = factor(weekday),
    workingday = factor(workingday),
    weathersit = factor(weathersit)
  )

cat("\nFinal dataset dimensions:\n")
print(dim(df))

cat("\nMissing values:\n")
print(colSums(is.na(df)))

cat("\nDuplicate rows:\n")
print(sum(duplicated(df)))

cat("\nDate range:\n")
print(range(df$date))

cat("\nUnique dates:\n")
print(n_distinct(df$date))

str(df)
summary(df)

write.csv(
  df,
  "data/bike_clean.csv",
  row.names = FALSE
)

cat("\nClean dataset saved to: data/bike_clean.csv\n")