# ============================================================
# Bike Rental Demand Forecasting
# 01 - Data Preparation
# ============================================================

# Packages ---------------------------------------------------

library(tidyverse)
library(caret)
library(lubridate)


# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

df_raw <- read.csv("data/bike_sharing.csv")

# Inspect dimensions
dim(df_raw)

# Inspect structure
str(df_raw)

# Preview data
head(df_raw)


# ------------------------------------------------------------
# 2. Create Date Variable
# ------------------------------------------------------------

df <- df_raw %>%
  mutate(
    date = as.Date(dteday)
  )


# ------------------------------------------------------------
# 3. Data Quality Checks
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Remove Incomplete Observations
# ------------------------------------------------------------

# Remove rows where the target variable is missing
df <- df %>%
  filter(!is.na(cnt))


# Verify missing values after removal
colSums(is.na(df))


# ------------------------------------------------------------
# 5. Remove Leakage / Identifier Variables
# ------------------------------------------------------------

# instant   = row identifier
# casual    = component of cnt
# registered = component of cnt
# dteday    = original character date; date is retained instead

df <- df %>%
  select(
    -instant,
    -dteday,
    -casual,
    -registered
  )


# ------------------------------------------------------------
# 6. Convert Categorical Variables to Factors
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. Final Data Quality Check
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8. Inspect Final Dataset
# ------------------------------------------------------------

str(df)

summary(df)


# ------------------------------------------------------------
# 9. Save Clean Dataset
# ------------------------------------------------------------

write.csv(
  df,
  "data/bike_clean.csv",
  row.names = FALSE
)

cat("\nClean dataset saved to: data/bike_clean.csv\n")