library(ggplot2)
install.packages("vcd")
library(vcd)     # For assocstats (Cramér's V)
library(dplyr)
library(tidyr)

# Load your data
df <- read.csv("training_data.csv")
target <- "risk_level"

# Convert low-cardinality numeric variables to factors
df <- df %>%
  mutate(across(where(~ is.numeric(.) && n_distinct(.) <= 10), as.factor))

# Ensure target is a factor
df[[target]] <- as.factor(df[[target]])

# Identify categorical variables (excluding target)
cat_vars <- names(df)[sapply(df, is.factor)]
cat_vars <- setdiff(cat_vars, target)

# Open a PDF to save each heatmap on its own page
pdf("categorical_heatmaps.pdf", width = 8, height = 6)

for (var in cat_vars) {
  
  # Contingency table
  tbl <- table(df[[var]], df[[target]])
  
  # Chi-square test & Cramér's V
  chi <- suppressWarnings(chisq.test(tbl))
  cramer_v <- assocstats(tbl)$cramer
  
  # Convert to proportion table for heatmap
  prop_tbl <- prop.table(tbl, margin = 1)  # proportions per row
  prop_df <- as.data.frame(prop_tbl)
  names(prop_df) <- c("Var", "Target", "Proportion")
  
  # Stats annotation text
  stats_text <- paste0(
    "Chi² = ", round(chi$statistic, 2), "\n",
    "df = ", chi$parameter, "\n",
    "p = ", signif(chi$p.value, 4), "\n",
    "Cramér's V = ", round(cramer_v, 3)
  )
  
  # Heatmap plot
  p <- ggplot(prop_df, aes(x = Target, y = Var, fill = Proportion)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", Proportion)), color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(
      title = paste0("Proportional Heatmap: '", var, "' vs '", target, "'"),
      x = target, y = var, fill = "Proportion"
    ) +
    theme_minimal(base_size = 14) +
    annotate("text", x = length(levels(df[[target]])) + 0.8,
             y = length(levels(df[[var]])) + 0.5,
             label = stats_text, hjust = 0, vjust = 1,
             size = 4.2, fontface = "bold")
  
  # Print plot to PDF
  print(p)
}

# Close PDF file
dev.off()

cat("✅ Saved as 'categorical_heatmaps.pdf' — each variable on its own page.\n")
