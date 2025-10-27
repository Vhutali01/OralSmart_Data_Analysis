#install.packages("ggplot2")


library(ggplot2)

training_data <- read.csv("training_data.csv")

# Get numeric variables except total_dmft_score
numeric_vars <- names(training_data)[sapply(training_data, is.numeric)]
numeric_vars <- setdiff(numeric_vars, "total_dmft_score")  # exclude the outcome

# Loop and plot
for (var in numeric_vars) {
  p <- ggplot(training_data, aes_string(x = var, 
                                        y = "total_dmft_score", 
                                        color = "risk_level")) +
    geom_point(size = 3) +
    labs(title = paste(var, "vs DMFT Score"),
         x = var,
         y = "DMFT Score") +
    theme_minimal()
  
  #print(p)  # shows plot
  
  ggsave(paste0("plot_", var, ".png"), plot = p, width = 6, height = 4)
}

