import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("global_gdp_inflation_2000_2024.csv")
df = pd.DataFrame(data)
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Filling missing values with the mean of the respective columns
df["GDP"] = df["GDP_Growth_Percent"].fillna(df["GDP_Growth_Percent"].mean())
df["Inflation"] = df["Inflation_Percent"].fillna(df["Inflation_Percent"].mean())

# Convert "Year" to datetime format
df["Year"] = pd.to_datetime(df["Year"], format="%Y")

# Ployying GDP and Inflation over the years
plt.figure(figsize=(12, 6))
sns.lineplot(x="Year", y="GDP", data=df, label="GDP")
sns.lineplot(x="Year", y="Inflation", data=df, label="Inflation")
plt.title("Global GDP and Inflation (2000-2024)")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.show()

# GDP trend over time
import matplotlib.pyplot as plt
gdp_trend = df.groupby('Year')['GDP_Growth_Percent'].mean()
plt.plot(gdp_trend)
plt.title("Average Global GDP Trend")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()

# Inflation Trend over time
inflation_trend = df.groupby('Year')['Inflation'].mean()
plt.plot(inflation_trend)
plt.title("Average Global Inflation Trend")
plt.xlabel("Year")
plt.ylabel("Inflation Percent")
plt.show()

# GDP vs Inflation Correlation
df[['GDP_Growth_Percent','Inflation_Percent']].corr()
sns.scatterplot(x="GDP_Growth_Percent", y="Inflation_Percent", data=df)
plt.title("GDP vs Inflation")
plt.xlabel("GDP Growth Percent")
plt.ylabel("Inflation Percent")
plt.show()

# Top & Bottom Countries Analysis
top_gdp = df.groupby('Country')['GDP'].mean().sort_values(ascending=False).head(10)
bottom_gdp = df.groupby('Country')['GDP'].mean().sort_values(ascending=True).head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_gdp.values, y=top_gdp.index, palette="viridis")
plt.title("Top 10 Countries by Average GDP")
plt.xlabel("Average GDP")
plt.ylabel("Country")
plt.show()
plt.figure(figsize=(8, 6))
sns.barplot(x=bottom_gdp.values, y=bottom_gdp.index, palette="viridis")
plt.title("Bottom 10 Countries by Average GDP")
plt.xlabel("Average GDP")
plt.ylabel("Country")
plt.show()

# GDP Growth Rate Calculation
df['GDP_Growth'] = df.groupby('Country')['GDP'].pct_change() * 100
# Inflation Impact on GDP Growth
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Inflation_Percent", y="GDP_Growth", data=df)
plt.title("Inflation vs GDP Growth")
plt.xlabel("Inflation Percent")
plt.ylabel("GDP Growth Percent")
plt.show()

# Rolling Average (Trend Smoothing)
df_year = df.groupby('Year')['GDP'].mean()
rolling_avg = df_year.rolling(window=3).mean()
plt.plot(df_year, label="Original")
plt.plot(rolling_avg, label="3-Year Moving Average")
plt.legend()
plt.title("Global GDP with Rolling Average")
plt.xlabel("Year")
plt.ylabel("Average GDP")
plt.show()

# Crisis Period Analysis (Business-Level Insight)
crisis_years = [2008, 2020]
for year in crisis_years:
    crisis_data = df[df["Year"].dt.year == year]
    plt.figure(figsize=(10, 5))
    plt.plot(crisis_data["Year"], crisis_data["GDP"], label="GDP")
    plt.title(f"GDP Trend in {year}")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.legend()
    plt.show()
# Insight:
# 1. Global GDP and Inflation have show significant fluctuations over the years, with notable dips during crisis periods.
# 2. There is a moderate correlation between GDP growth and inflation, suggesting that higher inflation may be associated with lower GDP growth.
# 3. The top countries by average GDP consistently outperform the bottom countries, indicating disparities in economic performance.
# 4. The rolling average helps to smooth out short-term fluctuations and highlight long-term trends in GDP.

# Saved Cleaned dataset
df.to_csv("cleaned_global_gdp_inflation_2000_2024.csv", index=False)
print("Cleaned dataset saved as 'cleaned_global_gdp_inflation_2000_2024.csv'")
