import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load historical data
data = pd.read_csv('~/Downloads/yarn_queue_utilization.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Prepare data for linear regression
X = np.array(range(len(data))).reshape(-1, 1)
y = data['utilization'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Project future utilization
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365)
future_X = np.array(range(len(data), len(data) + 365)).reshape(-1, 1)
future_y = model.predict(future_X)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['utilization'], label='Historical')
plt.plot(future_dates, future_y, label='Projected')
plt.title('YARN Resource Utilization - Historical and Projected')
plt.xlabel('Date')
plt.ylabel('Utilization (%)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate required capacity for next year with 20% headroom
max_projected_utilization = max(future_y)
required_capacity = max_projected_utilization * 1.2

print(f"Projected max utilization: {max_projected_utilization:.2f}%")
print(f"Recommended capacity with headroom: {required_capacity:.2f}%")
