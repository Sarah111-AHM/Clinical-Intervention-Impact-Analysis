library(tidyverse)
library(ggplot2)
library(plotly)
library(gridExtra)

data <- readRDS("data_clean.rds")

if (!dir.exists("output/figures")) dir.create("output/figures", recursive = TRUE)

p1 <- ggplot(data, aes(x = age, fill = group)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 15, color = "black") +
  labs(title = "Age distribution by group", x = "Age (years)", y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("Control" = "#E69F00", "Intervention" = "#56B4E9"))

ggsave("output/figures/age_distribution.png", p1, width = 6, height = 4)

ggplotly(p1)

p2 <- ggplot(data, aes(x = group, y = hospital_stay, fill = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(aes(color = complication), width = 0.2, alpha = 0.6) +
  labs(title = "Hospital stay by group and complication", x = "Group", y = "Hospital stay (days)") +
  theme_minimal() +
  scale_fill_manual(values = c("Control" = "#E69F00", "Intervention" = "#56B4E9")) +
  scale_color_manual(values = c("No" = "darkgreen", "Yes" = "red"))

ggsave("output/figures/hospital_stay_boxplot.png", p2, width = 6, height = 4)
ggplotly(p2)

comp_summary <- data %>%
  group_by(group, sex, complication) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(group, sex) %>%
  mutate(prop = count / sum(count) * 100) %>%
  filter(complication == "Yes")

p3 <- ggplot(comp_summary, aes(x = group, y = prop, fill = sex)) +
  geom_col(position = position_dodge(), width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", prop)),
            position = position_dodge(0.7), vjust = -0.5) +
  labs(title = "Complication rate by group and sex", x = "Group", y = "Complication rate (%)") +
  theme_minimal() +
  scale_fill_manual(values = c("Male" = "#0072B2", "Female" = "#CC79A7"))

ggsave("output/figures/complication_by_group_sex.png", p3, width = 6, height = 4)
ggplotly(p3)

p4 <- ggplot(data, aes(x = age, y = hospital_stay, color = group, shape = complication)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +
  labs(title = "Age vs hospital stay", x = "Age", y = "Hospital stay") +
  theme_minimal() +
  scale_color_manual(values = c("Control" = "#E69F00", "Intervention" = "#56B4E9")) +
  facet_wrap(~ group)

ggsave("output/figures/age_stay_scatter.png", p4, width = 8, height = 4)
ggplotly(p4)

heatmap_data <- data %>%
  group_by(age_group, stay_cat) %>%
  summarise(count = n(), .groups = "drop")

p5 <- ggplot(heatmap_data, aes(x = age_group, y = stay_cat, fill = count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = count), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Heatmap of age groups and stay categories",
       x = "Age group", y = "Stay category") +
  theme_minimal()

ggsave("output/figures/heatmap_age_staycat.png", p5, width = 6, height = 4)

if (file.exists("output/tables/glm_model.rds")) {
  glm_model <- readRDS("output/tables/glm_model.rds")
  pred_prob <- predict(glm_model, type = "response")
  roc_curve <- pROC::roc(data$complication, pred_prob, levels = c("No", "Yes"))
  png("output/figures/roc_curve.png")
  plot(roc_curve, col = "blue", main = "ROC curve - logistic model")
  dev.off()
}

combined <- grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)
ggsave("output/figures/combined_plots.png", combined, width = 12, height = 8)

cat("\nAll plots created and saved in output/figures/\n")
