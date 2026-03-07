library(tidyverse)
library(car)
library(lmtest)
library(sandwich)

data <- readRDS("data_clean.rds")

shapiro_age <- shapiro.test(data$age)
shapiro_stay <- shapiro.test(data$hospital_stay)

cat("\n========== Shapiro-Wilk Normality Test ==========\n")
cat("Age: W =", shapiro_age$statistic, ", p-value =", shapiro_age$p.value, "\n")
cat("Hospital stay: W =", shapiro_stay$statistic, ", p-value =", shapiro_stay$p.value, "\n")

cat("\n========== Age comparison between groups ==========\n")
if (shapiro_age$p.value > 0.05) {
  t_test_age <- t.test(age ~ group, data = data, var.equal = TRUE)
  print(t_test_age)
} else {
  wilcox_age <- wilcox.test(age ~ group, data = data)
  print(wilcox_age)
}

cat("\n========== Hospital stay comparison between groups ==========\n")
if (shapiro_stay$p.value > 0.05) {
  t_test_stay <- t.test(hospital_stay ~ group, data = data, var.equal = TRUE)
  print(t_test_stay)
} else {
  wilcox_stay <- wilcox.test(hospital_stay ~ group, data = data)
  print(wilcox_stay)
}

cat("\n========== Chi-square test: Group × Complication ==========\n")
tbl_group_comp <- table(data$group, data$complication)
chi_test <- chisq.test(tbl_group_comp, correct = TRUE)
print(chi_test)

if (all(dim(tbl_group_comp) == 2)) {
  OR <- (tbl_group_comp[1,1] * tbl_group_comp[2,2]) / (tbl_group_comp[1,2] * tbl_group_comp[2,1])
  cat("\nOdds Ratio (OR) for complications in Intervention vs Control:", round(OR, 2), "\n")
  OR_ci <- fisher.test(tbl_group_comp)$conf.int
  cat("95% Confidence Interval: [", round(OR_ci[1], 2), ",", round(OR_ci[2], 2), "]\n")
}

cat("\n========== Two-way ANOVA (group * sex) for hospital stay ==========\n")
anova_2way <- aov(hospital_stay ~ group * sex, data = data)
summary(anova_2way)

leveneTest(hospital_stay ~ group * sex, data = data)

cat("\n========== Hospital stay comparison by group inside each sex ==========\n")
wilcox.test(hospital_stay ~ group, data = subset(data, sex == "Male"))
wilcox.test(hospital_stay ~ group, data = subset(data, sex == "Female"))

cat("\n========== Chi-square test: Sex × Complication ==========\n")
tbl_sex_comp <- table(data$sex, data$complication)
chi_sex <- chisq.test(tbl_sex_comp)
print(chi_sex)

sink("output/tables/inferential_tests_results.txt")
cat("Statistical Test Results:\n\n")
cat("Shapiro-Wilk:\n"); print(shapiro_age); print(shapiro_stay)
cat("\nAge comparison:\n"); if (exists("t_test_age")) print(t_test_age) else print(wilcox_age)
cat("\nHospital stay comparison:\n"); if (exists("t_test_stay")) print(t_test_stay) else print(wilcox_stay)
cat("\nChi-square group x complication:\n"); print(chi_test)
cat("\nTwo-way ANOVA:\n"); summary(anova_2way)
sink()

cat("\nStatistical tests completed.\n")
