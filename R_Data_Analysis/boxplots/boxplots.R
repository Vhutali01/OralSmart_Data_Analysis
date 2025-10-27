#install.packages("ggplot2")

library(ggplot2)

training_data <- read.csv("training_data.csv")

# Get categorical variables except the outcome
categorical_vars <- names(training_data)[sapply(training_data, is.numeric)]
categorical_vars <- setdiff(categorical_vars, "risk_level")  # optional exclusion

# Loop and plot
for (var in categorical_vars) {
  p <- ggplot(training_data, aes_string(x = var, 
                                        y = "total_dmft_score", 
                                        fill = "risk_level")) +
    geom_boxplot() +
    labs(title = paste(var, "vs DMFT Score"),
         x = var,
         y = "DMFT Score") +
    theme_minimal()
  
  # Save each plot
  ggsave(paste0("boxplot_", var, ".png"), plot = p, width = 6, height = 4)
}
