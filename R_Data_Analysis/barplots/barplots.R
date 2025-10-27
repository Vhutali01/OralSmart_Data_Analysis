# Install/load ggplot2 if needed
install.packages("ggplot2")
library(ggplot2)

# Read data
training_data <- read.csv("training_data.csv")

# Ensure risk_level is a factor with fixed order
training_data$risk_level <- factor(training_data$risk_level, 
                                   levels = c("low", "medium", "high"))

# Define consistent colors for risk levels
risk_colors <- c("low" = "#1f78b4",    # blue
                 "medium" = "#33a02c", # green
                 "high" = "#e31a1c")   # red

# Get variables to plot (excluding outcome and risk_level itself)
vars_to_plot <- setdiff(names(training_data), 
                        c("total_dmft_score", "risk_level"))

# Loop through variables
for (var in vars_to_plot) {
  
  # Only plot if variable has <= 10 unique values (categorical-like)
  if (length(unique(training_data[[var]])) <= 10) {
    
    p <- ggplot(training_data, 
                aes(x = as.factor(.data[[var]]), 
                    fill = risk_level)) +
      geom_bar(position = position_dodge(width = 0.8)) +
      labs(title = paste(var, "(Barplot) by Risk Level"),
           x = var,
           y = "Count") +
      scale_fill_manual(values = risk_colors) +
      theme_minimal()
    
    # Save image
    ggsave(paste0("barplot_", var, ".png"), plot = p, width = 6, height = 4)
  }
}
