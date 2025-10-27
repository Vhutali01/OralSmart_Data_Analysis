library(dplyr)
library(ggplot2)

training_data <- read.csv("training_data.csv")

# Target variable
target_var <- "risk_level"

# Convert numeric variables with <=10 unique values to factors
training_data <- training_data %>%
  mutate(across(where(is.numeric), ~ if (n_distinct(.) <= 10) factor(.) else .))

# Identify categorical variables (excluding target)
categorical_vars <- setdiff(
  names(training_data)[sapply(training_data, is.factor)],
  target_var
)

library(gridExtra)

for (var in categorical_vars) {
  
  # Create contingency table
  tbl <- table(training_data[[var]], training_data[[target_var]])
  
  # Skip if the table is degenerate (zero counts or only one category)
  if (min(dim(tbl)) < 2) next
  
  # Perform Chi-square test
  chi_res <- chisq.test(tbl)
  
  # Prepare results text
  results_text <- paste0(
    "Chi-square Test: ", var, " vs ", target_var, "\n",
    "Chi-sq = ", round(chi_res$statistic, 3), 
    ", df = ", chi_res$parameter, 
    ", p-value = ", signif(chi_res$p.value, 3)
  )
  
  # Create a plot showing observed counts
  p <- ggplot(as.data.frame(tbl), aes(x = Var1, y = Freq, fill = Var2)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
      title = paste("Chi-square Test:", var, "vs", target_var),
      subtitle = results_text,
      x = var,
      y = "Count",
      fill = target_var
    ) +
    theme_minimal()
  
  # Save the plot
  ggsave(paste0("chi_square_", var, ".png"), plot = p, width = 7, height = 5)
}

