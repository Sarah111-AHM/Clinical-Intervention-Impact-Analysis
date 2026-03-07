library(tidyverse)
data <- read.csv("competition_dataset.csv", header = TRUE, stringsAsFactors = FALSE)
glimpse(data)
data <- data %>%
  mutate(
    sex = factor(sex, levels = c("Male", "Female")),
    group = factor(group, levels = c("Control", "Intervention")),
    complication = factor(complication, levels = c("No", "Yes")),
    age_group = cut(age, breaks = c(0, 40, 60, 100),
                    labels = c("Young (<40)", "Middle (40-60)", "Old (>60)"),
                    right = FALSE),
    stay_cat = cut(hospital_stay, breaks = c(0, 3, 7, 30),
                   labels = c("Short (<=3)", "Medium (4-7)", "Long (>=8)"),
                   right = FALSE)
  )

if (anyNA(data)) {
  cat("Missing values found. Cleaning data.\n")
  data <- na.omit(data)
} else {
  cat("No missing values. Data is clean.\n")
}
saveRDS(data, file = "data_clean.rds")
cat("Data loaded and cleaned. Rows:", nrow(data), "\n")
