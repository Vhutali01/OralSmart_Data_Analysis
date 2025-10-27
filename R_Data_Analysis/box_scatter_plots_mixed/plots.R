# Install/load libraries
install.packages("ggplot2")
library(ggplot2)

# Read data
training_data <- read.csv("training_data.csv")

# Ensure risk_level is a factor with a fixed order
training_data$risk_level <- factor(training_data$risk_level, 
                                   levels = c("low", "medium", "high"))

# Define consistent colors for risk levels
risk_colors <- c("low" = "#1f78b4",    # blue
                 "medium" = "#33a02c", # green
                 "high" = "#e31a1c")   # red

# Get all numeric variables except the outcome
numeric_vars <- setdiff(names(training_data)[sapply(training_data, is.numeric)], 
                        "total_dmft_score")

# Loop and plot each variable
for (var in numeric_vars) {
  
  # If variable has few unique values (categorical-like)
  if (length(unique(training_data[[var]])) <= 5) {
    
    p <- ggplot(training_data, 
                aes(x = as.factor(.data[[var]]), 
                    y = total_dmft_score, 
                    fill = risk_level)) +
      geom_boxplot(position = position_dodge(width = 0.8)) +
      labs(title = paste(var, "(Boxplot) vs DMFT Score"),
           x = var,
           y = "DMFT Score") +
      scale_fill_manual(values = risk_colors) +
      theme_minimal()
    
    ggsave(paste0("boxplot_", var, ".png"), plot = p, width = 6, height = 4)
    
  } else {
    # Continuous variable → scatter plot
    p <- ggplot(training_data, 
                aes(x = .data[[var]], 
                    y = total_dmft_score, 
                    color = risk_level)) +
      geom_point(size = 2, alpha = 0.7) +
      labs(title = paste(var, "(Scatter) vs DMFT Score"),
           x = var,
           y = "DMFT Score") +
      scale_color_manual(values = risk_colors) +
      theme_minimal()
    
    ggsave(paste0("scatter_", var, ".png"), plot = p, width = 6, height = 4)
  }
}

