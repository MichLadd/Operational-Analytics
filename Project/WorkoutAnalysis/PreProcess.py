import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./WorkoutData/workout_dataset_split.csv', sep=",")

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Remove rows with NaT values in the 'Date' column
df.dropna(subset=['Date'], inplace=True)

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Convert the 'Weight (kg)', 'Sets', 'Reps' columns to numeric
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Sets'] = pd.to_numeric(df['Sets'], errors='coerce').fillna(4)
df['Reps'] = pd.to_numeric(df['Reps'], errors='coerce').fillna(12)

# Remove rows with NaN values
df.dropna(subset=['Weight', 'Sets', 'Reps'], inplace=True)

# Calculate the tonnage (total weight lifted) for each exercise
df['tonnage'] = df['Weight'] * df['Sets'] * df['Reps']

# Weekly aggregation of tonnage by muscle group
weekly_data = df.groupby([pd.Grouper(freq='W'), 'Exercise']).agg({'tonnage': 'sum'}).reset_index()

# Pivot to have muscle groups as columns
weekly_data_pivot = weekly_data.pivot(index='Date', columns='Exercise', values='tonnage').fillna(0)

# Save the preprocessed dataset
weekly_data_pivot.to_csv('./WorkoutData/preprocessed_data.csv')

# Plot the trend of tonnage by muscle group to get an idea of the workload distribution
'''
plt.figure(figsize=(12, 6))
for muscle_group in weekly_data_pivot.columns:
    plt.plot(weekly_data_pivot.index, weekly_data_pivot[muscle_group], label=muscle_group)
    # plt.title('Andamento del tonnellaggio per {}'.format(muscle_group))
    # plt.xlabel('Data')
    # plt.ylabel('Tonnellaggio')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

plt.title('Andamento del tonnellaggio per gruppo muscolare')
plt.xlabel('Data')
plt.ylabel('Tonnellaggio')
plt.legend()
plt.grid(True)
plt.show()
'''