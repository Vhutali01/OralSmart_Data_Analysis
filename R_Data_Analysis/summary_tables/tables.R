library(dplyr)

# Identify categorical vars: factors or character
categorical_vars <- names(training_data)[sapply(training_data, function(x) is.factor(x) || is.character(x))]


summary_tables <- list()

for (var in categorical_vars) {
  summary_df <- training_data %>%
    group_by(across(all_of(var))) %>%
    summarise(
      Frequency = n(),
      Min_DMFT = min(total_dmft_score, na.rm = TRUE),
      Max_DMFT = max(total_dmft_score, na.rm = TRUE),
      Median_DMFT = median(total_dmft_score, na.rm = TRUE),
      Mean_DMFT = mean(total_dmft_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(across(all_of(var)))
  
  summary_tables[[var]] <- summary_df
}

# To see summary for a particular variable, e.g. "risk_level":
print(summary_tables[["risk_level"]])


combined_summary <- bind_rows(
  lapply(names(summary_tables), function(var) {
    summary_tables[[var]] %>%
      mutate(Variable = var) %>%
      rename(Category = !!sym(var))
  })
)

print(combined_summary)
