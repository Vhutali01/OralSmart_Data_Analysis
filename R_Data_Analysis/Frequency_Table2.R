
training_data <- read.csv("training_data.csv", header = TRUE, stringsAsFactors = FALSE)

head(training_data)

lapply(training_data, table)

tables_by_risk <- lapply(training_data[ , names(training_data) != "risk_level"], function(x) {
  table(x, training_data$risk_level)
})

lapply(tables_by_risk, table)
