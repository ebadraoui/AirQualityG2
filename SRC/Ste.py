import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV = "C:/Users/elyes/Downloads/beijing+pm2+5+data (1)/PRSA_data_2010.1.1-2014.12.31.csv"

df = pd.read_csv(CSV)
df = df[df['hour'].between(0,23)].copy()

df['healthy'] = np.where(df['pm2.5'] < 75, 1, 0)

df['hour_sin']  = np.sin(2*np.pi*df['hour']/24.0)
df['hour_cos']  = np.cos(2*np.pi*df['hour']/24.0)
df['month_sin'] = np.sin(2*np.pi*df['month']/12.0)
df['month_cos'] = np.cos(2*np.pi*df['month']/12.0)

print("Rows:", len(df))
print("Columns:", df.shape[1])
print("Missing pm2.5:", df['pm2.5'].isna().sum())

df_clf = df.dropna(subset=['pm2.5']).copy()
df_reg = df.dropna(subset=['pm2.5']).copy()



counts = df_clf['healthy'].value_counts().sort_index()
plt.figure(figsize=(6,4))
sns.barplot(x=['Unhealthy (0)', 'Healthy (1)'], y=counts.values)
plt.title("Plot 1: Target Distribution (Healthy vs Unhealthy)")
plt.ylabel("Count of samples")
plt.tight_layout()
plt.savefig("plot1_target_distribution.png")
plt.show()

num_cols = ['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir','hour','month']
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Plot 2: Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png")
plt.show()

