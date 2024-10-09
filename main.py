import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets (update the paths as necessary)
bookings_df = pd.read_csv('/Users/snehalpatil/PycharmProjects/myproject1/Book.csv')
sessions_df = pd.read_csv('/Users/snehalpatil/PycharmProjects/myproject1/sessions.csv')

# Merge datasets on 'booking_id'
merged_df = pd.merge(bookings_df, sessions_df, how='right', on='booking_id', suffixes=('_bookings', '_sessions'))

# Ensure 'booking_time' is converted to datetime and strip timezone if present
bookings_df['booking_time'] = pd.to_datetime(bookings_df['booking_time']).dt.tz_localize(None)  # Remove timezone info
bookings_df['booking_date'] = bookings_df['booking_time'].dt.date  # Extract date
bookings_df['day_of_week'] = bookings_df['booking_time'].dt.day_name()  # Extract day of the week

# Ensure 'day_of_week' exists in the merged dataframe
merged_df['booking_time'] = pd.to_datetime(merged_df['booking_time']).dt.tz_localize(None)  # Convert again after merge if needed
merged_df['day_of_week'] = merged_df['booking_time'].dt.day_name()  # Extract day of the week again

# 1. Find the number of distinct bookings, sessions, and searches
distinct_bookings = merged_df['booking_id'].nunique()
distinct_sessions = merged_df['session_id'].nunique()
distinct_searches = merged_df['search_id'].nunique()

print(f'Distinct Bookings: {distinct_bookings}')
print(f'Distinct Sessions: {distinct_sessions}')
print(f'Distinct Searches: {distinct_searches}')

# 2. Sessions with more than one booking
multi_booking_sessions = merged_df.groupby('session_id').size()
sessions_with_multiple_bookings = multi_booking_sessions[multi_booking_sessions > 1].count()

print(f'Sessions with more than one booking: {sessions_with_multiple_bookings}')

# 3. Days of the week with the highest number of bookings
daywise_bookings = bookings_df['day_of_week'].value_counts()
print(f'Days of the week with highest bookings:\n{daywise_bookings}')

# Pie chart for bookings distribution by day of the week
daywise_bookings.plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('Bookings Distribution by Day of the Week')
plt.show()

# 4. Total number of bookings and Gross Booking Value in INR for each service
service_summary = bookings_df.groupby('service_name').agg({
    'booking_id': 'count',
    'INR_Amount': 'sum'
}).reset_index()
service_summary.columns = ['Service Name', 'Total Bookings', 'Total Gross Booking Value (INR)']
print(service_summary)

# 5. Most booked route (from_city to to_city) for customers with more than 1 booking
customer_bookings = bookings_df.groupby(['customer_id', 'from_city', 'to_city']).size().reset_index(name='counts')
customer_bookings = customer_bookings[customer_bookings['counts'] > 1]
most_booked_route = customer_bookings.groupby(['from_city', 'to_city'])['counts'].sum().idxmax()

print(f'Most booked route: {most_booked_route}')

# 6. Top 3 departure cities for advance bookings
advance_bookings = bookings_df[bookings_df['days_to_departure'] >= 5]
top_departure_cities = advance_bookings['from_city'].value_counts().nlargest(3)

print(f'Top 3 departure cities with advance bookings:\n{top_departure_cities}')

# 7. Heatmap of correlations of numerical columns
numerical_cols = bookings_df.select_dtypes(include=['number'])
correlation_matrix = numerical_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Maximum correlation pair
max_corr_pair = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().iloc[1]
print(f'Maximum correlation pair: {max_corr_pair}')

# 8. Most used device type for each service
device_usage = bookings_df.groupby(['service_name', 'device_type_used']).size().unstack().idxmax(axis=1)
print(f'Most used device type for each service:\n{device_usage}')

# 9. Quarterly trends of bookings by device type
merged_df['quarter'] = pd.to_datetime(merged_df['booking_time']).dt.to_period('Q')
quarterly_device_trends = merged_df.groupby(['quarter', 'device_type_used'])['booking_id'].count().unstack()

# Plot the quarterly trends of bookings by device type
quarterly_device_trends.plot(figsize=(12, 6))
plt.title('Quarterly Trends of Bookings by Device Type')
plt.xlabel('Quarter')
plt.ylabel('Number of Bookings')
plt.legend(title='Device Type')
plt.show()

# 10. Overall Booking to Search Ratio (oBSR) analysis
# Monthly oBSR
merged_df['month'] = pd.to_datetime(merged_df['booking_time']).dt.month
monthly_oBSR = merged_df.groupby('month').agg({'search_id': 'count', 'booking_id': 'count'})
monthly_oBSR['oBSR'] = monthly_oBSR['booking_id'] / monthly_oBSR['search_id']
print(f'Average oBSR per month: {monthly_oBSR["oBSR"].mean()}')

# Daily oBSR
daily_oBSR = merged_df.groupby('day_of_week').agg({'search_id': 'count', 'booking_id': 'count'})
daily_oBSR['oBSR'] = daily_oBSR['booking_id'] / daily_oBSR['search_id']
print(f'Average oBSR per day of the week: {daily_oBSR["oBSR"].mean()}')

# Time Series oBSR
merged_df['date'] = pd.to_datetime(merged_df['booking_time']).dt.date
datewise_oBSR = merged_df.groupby('date').agg({'search_id': 'count', 'booking_id': 'count'})
datewise_oBSR['oBSR'] = datewise_oBSR['booking_id'] / datewise_oBSR['search_id']

# Plot time series of oBSR
plt.figure(figsize=(12, 6))
plt.plot(datewise_oBSR.index, datewise_oBSR['oBSR'])
plt.title('Time Series of oBSR')
plt.xlabel('Date')
plt.ylabel('oBSR')
plt.show()
