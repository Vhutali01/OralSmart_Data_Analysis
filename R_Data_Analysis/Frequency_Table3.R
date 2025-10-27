
training_data <- read.csv("training_data.csv", header = TRUE, stringsAsFactors = FALSE)


#install.packages("purrr")
#install.packages("dplyr")
#install.packages("knitr")
#install.packages("tidyr")


library(dplyr)
library(tidyr)
library(knitr)

# Replace 'training_data' with your actual data frame name
# Ensure 'risk_level' is a column in your data with values like 'low', 'medium', 'high'

# Get variable names except the target
vars <- setdiff(names(training_data), "risk_level")

# Create summary table
summary_table <- lapply(vars, function(var) {
  training_data %>%
    group_by(!!sym(var), risk_level) %>%
    summarise(n = n(), .groups = 'drop') %>%
    group_by(!!sym(var)) %>%
    mutate(percentage = round(100 * n / sum(n), 1)) %>%
    unite("value_percent", n, percentage, sep = ", ") %>%
    pivot_wider(names_from = risk_level, values_from = value_percent) %>%
    mutate(Variable = var) %>%
    rename(Value = !!sym(var)) %>%
    select(Variable, Value, low, medium, high)
}) %>%
  bind_rows()

# Replace NA with blank for cleaner display
summary_table[is.na(summary_table)] <- ""

# Optional: rename risk columns to make them more presentable
colnames(summary_table) <- c("Variable", "Value", "Low Risk (n,%)", "Medium Risk (n,%)", "High Risk (n,%)")

# Print as a pretty table
kable(summary_table, caption = "Risk Level Breakdown by Variable (Counts and Percentages)")


install.packages("gridExtra")
install.packages("grid")


# Load additional libraries
library(gridExtra)
library(grid)

# Create a table graphic object (grob)
table_grob <- tableGrob(summary_table, rows = NULL)

# Save it as an image (PNG)
png("risk_summary_table.png", width = 1600, height = 1000, res = 150)
grid.draw(table_grob)
dev.off()

