# Dataset 1
options(stringsAsFactors = F)
library(xlsx)
library(tidyverse)
library(arules)
library(arulesViz)
library(RColorBrewer)  # color palettes for plots
library(shinythemes)   
library(readr)
library(readxl)
library("arulesViz")
library(RColorBrewer)

# Import data
Mart <- read_excel("Power Mart.xlsx")
new_data <- Mart[c("Order ID", "Product Name")]
new_data <- unique(new_data)
write.csv(new_data,"ts.csv", fileEncoding = 'GBK')
ts <- read.transactions("ts.csv", header = T, format = "single", cols = c(2, 3), sep = ",")
inspect(ts[1:15])
order <- read.csv("order_products_train.csv")
department <- read.csv("departments.csv")
aisles <- read.csv("aisles.csv")
product <- read.csv("products.csv")
order_data <- merge(order, product, by.x = "product_id", by.y = "product_id", all.x = TRUE, all.y = FALSE)

# Apriori Algorithm
rules <- apriori(ts, parameter = list(support = 0.0002, confidence = 0.3))
inspectDT(rules)
plot(rules)

write.csv(order_data[, c(2, 5)], "ultimate.csv", fileEncoding = "GBK")
ult <- read.transactions("ultimate.csv", format = "single", sep = ",", col = c(2, 3))
rules3 <- apriori(ult, parameter = list(support = 0.003, confidence = 0.3))
inspectDT(rules3)
plot(rules3, method = "graph")


# Explore rules according to region/subcategory
Mart <- read_excel("Power Mart.xlsx")
new_data <- Mart[c("Order ID", "Sub-Category")]
new_data <- unique(new_data)
write.csv(new_data, "sub.csv", fileEncoding = 'GBK')

groceries <- read.transactions("./sub.csv", format="single", sep = ',', col = c(2, 3))
summary(groceries)

groceryrules <- apriori(groceries, parameter = list(support = 0.001, confidence = 0.5, minlen = 1))
summary(groceryrules)
inspect(groceryrules[1:5])

plot(groceryrules, control = list(jitter = 2, col = rev(brewer.pal(9, "Greens")[4:9])), shading = "lift")
plot(groceryrules, method = "grouped", control = list(col = rev(brewer.pal(9, "Greens")[4:9])))

groceryrules_sort <- sort(groceryrules, by="support")
plot(groceryrules_sort[1:20], measure = "support", method = "graph", control = list(type = "items"), shading = "lift")

interestMeasure(groceryrules[1:5], measure = 'phi', transactions = groceries)

a <- merge(order, product, by.x = "product_id", by.y = "product_id", all.x = TRUE, all.y = FALSE)
b <- merge(a, department, by.x = "department_id", by.y = "department_id", all.x = TRUE, all.y = FALSE)
order_data <- merge(b, aisles, by.x = "aisle_id", by.y = "aisle_id", all.x = TRUE, all.y = FALSE)


order_data_1 <- order_data[, c(4, 8)]
c1 <- which(is.na(order_data_1$department))
order_data_1 <- order_data_1[-c1, ]
order_data_2 <- order_data[, c(4, 9)]

order_data_1 <- unique(order_data_1)
order_data_1 <- na.omit(order_data_1)
order_data_2 <- unique(order_data_2)
order_data_2 <- na.omit(order_data_2)

write_csv(order_data_1, "sub_department_1.csv", col_names = FALSE)
write_csv(order_data_2, "sub_department_2.csv", col_names = FALSE)
subdata_1 <- read.transactions("sub_department_1.csv", format = "single", sep = ",", col = c(1, 2))
subdata_2 <- read.transactions("sub_department_2.csv", format = "single", sep = ",", col = c(1, 2))

inspect(subdata_1[1:5])
inspect(subdata_2[1:5])

rules <- apriori(subdata_1, parameter = list(support = 0.1, confidence = 0.4))
inspectDT(rules)
plot(rules, control = list(jitter = 2, col = rev(brewer.pal(9, "Greens")[4:9])), shading = "lift")
plot(rules, method = "grouped", control = list(col = rev(brewer.pal(9, "Greens")[4:9])))
plot(rules[1:10], measure="confidence", method="graph", control = list(type = "items"), shading = "lift")

rules.sub <- subset(rules, subset = lift > 1.126)
plot(rules.sub, method = "graph", engine = "html")

rules2 <- apriori(subdata_2, parameter = list(support = 0.1, confidence = 0.4))
inspectDT(rules2)
plot(rules2, control = list(jitter = 2, col = rev(brewer.pal(9, "Greens")[4:9])), shading = "lift")
plot(rules2, method="grouped", control = list(col = rev(brewer.pal(9, "Greens")[4:9])))
plot(rules2[1:10], measure="confidence", method="graph", control = list(type = "items"), shading = "lift")
rules.sub2 <- subset(rules2, subset = support > 0.2 )
plot(rules.sub2, method = "graph", engine = "html")
