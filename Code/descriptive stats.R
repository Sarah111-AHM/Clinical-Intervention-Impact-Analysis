library(tidyverse)
library(gtsummary)
library(psych)
library(corrplot)
library(Hmisc)

data <- readRDS("data_clean.rds")

cat("\n================= General Statistics =================\n")
summary(data)

desc_stats <- data %>%
  group_by(group) %>%
  summarise(
    patients = n(),
    mean_age = mean(age, na.rm = TRUE),
    sd_age = sd(age, na.rm = TRUE),
    mean_stay = mean(hospital_stay, na.rm = TRUE),
    sd_stay = sd(hospital_stay, na.rm = TRUE),
    complication_rate = mean(complication == "Yes") * 100
  )
print(desc_stats)

tab_desc <- data %>%
  select(age, sex, group, hospital_stay, complication, age_group, stay_cat) %>%
  tbl_summary(
    by = group,
    statistic = list(
      all_continuous() ~ "{mean} ({sd}) [{min} - {max}]",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 1,
    label = list(
      age ~ "Age (years)",
      sex ~ "Sex",
      hospital_stay ~ "Hospital stay (days)",
      complication ~ "Complication",
      age_group ~ "Age group",
      stay_cat ~ "Stay category"
    )
  ) %>%
  add_p(test = list(all_continuous() ~ "wilcox.test", all_categorical() ~ "chisq.test")) %>%
  add_overall() %>%
  modify_header(label ~ "**Variable**") %>%
  modify_caption("**Table 1: Patient characteristics by group**") %>%
  bold_labels()

tab_desc

describeBy(data[, c("age", "hospital_stay")], group = data$group)

cor_matrix <- data %>%
  select(age, hospital_stay) %>%
  cor(method = "spearman")

corrplot(cor_matrix, method = "number", type = "upper", tl.col = "black",
         title = "Spearman Correlation Matrix")

cat("\n=== Cross table (Group × Sex × Complication) ===\n")
ftable(xtabs(~ group + sex + complication, data = data))

write.csv(desc_stats, "output/tables/desc_stats_by_group.csv", row.names = FALSE)

cat("\nDescriptive statistics completed.\n")
