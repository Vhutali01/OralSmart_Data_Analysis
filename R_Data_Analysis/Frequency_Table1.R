
training_data <- read.csv("training_data.csv", header = TRUE, stringsAsFactors = FALSE)


#install.packages("purrr")
#install.packages("dplyr")


library(purrr)
library(dplyr)


freq_tables <- map(training_data, ~ as.data.frame(table(.x)))


names(freq_tables) <- names(training_data)


for (col in names(freq_tables)){
  cat("\nColumn: ", col, "\n")
  print(freq_tables[[col]])
}

