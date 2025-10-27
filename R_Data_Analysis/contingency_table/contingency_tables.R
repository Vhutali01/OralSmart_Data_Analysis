library(ggplot2)
library(vcd)   # for assocstats
library(dplyr)
library(tidyr)

# Load data
df <- read.csv("training_data.csv")
target <- "risk_level"

# Convert numerically encoded categorical variables to factors
df <- df %>%
  mutate(across(where(~ is.numeric(.) && n_distinct(.) <= 10), as.factor))

# Ensure target is a factor
df[[target]] <- as.factor(df[[target]])

# Get categorical variables excluding target
cat_vars <- names(df)[sapply(df, is.factor)]
cat_vars <- setdiff(cat_vars, target)

# Open PDF to save all plots
pdf("chi_square_plots.pdf", width = 8, height = 6)

# Loop over categorical variables
for (var in cat_vars) {
  
  # Contingency table
  tbl <- table(df[[var]], df[[target]])
  
  # Chi-square and Cramér's V
  chi <- suppressWarnings(chisq.test(tbl)) # suppress warnings for small counts
  cramer_v <- assocstats(tbl)$cramer
  
  # Convert to data frame for ggplot
  tbl_df <- as.data.frame(tbl)
  names(tbl_df) <- c("Var", "Target", "Count")
  
  # Stats text for annotation
  stats_text <- paste0(
    "Chi² = ", round(chi$statistic, 2), "\n",
    "df = ", chi$parameter, "\n",
    "p = ", signif(chi$p.value, 4), "\n",
    "Cramér's V = ", round(cramer_v, 3)
  )
  
  # Plot
  p <- ggplot(tbl_df, aes(x = Target, y = Var, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(
      title = paste0("Chi-square test for '", var, "' vs '", target, "'"),
      x = target, y = var
    ) +
    theme_minimal() +
    annotate("text", x = 1.5, y = Inf, label = stats_text,
             hjust = 0, vjust = 1.1, size = 4, fontface = "bold",
             inherit.aes = FALSE)
  
  print(p)
}

# Close PDF
dev.off()

cat("PDF saved as chi_square_plots.pdf in your working directory.\n")


