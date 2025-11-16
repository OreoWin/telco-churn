#Setup

library(tidyverse)
library(janitor)

telco_raw = read.csv("C:/Users/lenovo/Desktop/Churn Project/WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(telco_raw)
summary(telco_raw)

#data cleaning 
any(is.na(telco_raw)) #check for missing value 
names(telco_raw)[colSums(is.na(telco_raw)) > 0] #target missing value 
sum(is.na(telco_raw$TotalCharges)) #11

telco = telco_raw %>%
  mutate(SeniorCitizen = if_else(SeniorCitizen == 1, "Yes","No"))%>% #covert numerical variables to binary
  na.omit() #delete records with NA since there's only 11 records 


numeric_vars <- telco %>%
  select(where(is.numeric)) %>%
  names()

cat_vars <- telco %>%
  select(-all_of(numeric_vars)) %>%
  names()


#churn rate 
churn_summary <- telco %>%
  count(Churn) %>%
  mutate(prop = n / sum(n))

overall_churn_rate <- churn_summary %>%
  filter(Churn == "Yes") %>%
  pull(prop)

overall_churn_rate #0.266

# Bar plot of churn distribution
ggplot(churn_summary, aes(x = Churn, y = prop)) +
  geom_col() +
  geom_text(aes(label = scales::percent(prop, accuracy = 0.1)),
            vjust = -0.3, size = 3) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Churn Distribution",
       x = "Churn",
       y = "Proportion") +
  theme_minimal()

#numerical variables: distribution 
telco_num_long <- telco %>%
  select(all_of(numeric_vars)) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "value")

#histogram
ggplot(telco_num_long, aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Distribution of Numeric Variables",
       x = NULL,
       y = "Count") +
  theme_minimal()

#boxplot 
telco_num_long_churn <- telco %>%
  select(Churn, all_of(numeric_vars)) %>%
  pivot_longer(cols = -Churn,
               names_to = "variable",
               values_to = "value")

ggplot(telco_num_long_churn, aes(x = Churn, y = value)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Numeric Variables by Churn",
       x = "Churn",
       y = NULL) +
  theme_minimal()

#numerical summary by churn 
numeric_summary_by_churn <- telco %>%
  group_by(Churn) %>%
  summarise(
    across(
      .cols = all_of(numeric_vars),
      .fns  = list(mean = ~ mean(.x, na.rm = TRUE),
                   median = ~ median(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    )
  )

numeric_summary_by_churn

#categorical variables 
cat_vars_plot <- setdiff(cat_vars, c("customerID")) 
cat_vars_plot <- setdiff(cat_vars_plot, "Churn")     # target separate

plot_cat_by_churn <- function(df, var) {
  ggplot(df, aes(x = .data[[var]], fill = Churn)) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      title = paste("Churn Rate by", var),
      x = var,
      y = "Proportion"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

key_cat_vars <- c("Contract", "InternetService", "PaymentMethod",
                  "OnlineSecurity", "TechSupport")

for (v in key_cat_vars) {
  if (v %in% names(telco)) {
    print(plot_cat_by_churn(telco, v))
  }
}

#loop through all categorical variables:
for (v in cat_vars_plot) {
  print(plot_cat_by_churn(telco, v))
}

telco$churn_binary = ifelse(telco$Churn == 'Yes',1,0)%>% as.integer()

write.csv(telco,
          file = "telco_cleaned.csv",
          row.names = FALSE)
