import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the historical data
# Assuming you have a CSV file with columns 'ds' (date) and 'y' (utilization percentage)
df = pd.read_csv('~/Downloads/yarn_queue_utilization.csv')
#df = pd.read_csv('yarn_queue_utilization.csv')

# Initialize the Prophet model
model = Prophet()

# Fit the model to the historical data
model.fit(df)

# Create a dataframe with future dates
future_dates = model.make_future_dataframe(periods=10)  # Forecast for the next year

# Make predictions
forecast = model.predict(future_dates)

# Plot the forecast
fig1 = model.plot(forecast)
plt.title('YARN Queue Capacity Forecast')
plt.xlabel('Date')
plt.ylabel('Utilization Percentage')
plt.show()

# Plot the components of the forecast
fig2 = model.plot_components(forecast)
plt.show()

# Print the forecast for the next 30 days
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# Save the forecast to a CSV file
forecast.to_csv('/tmp/yarn_queue_forecast.csv', index=False)
