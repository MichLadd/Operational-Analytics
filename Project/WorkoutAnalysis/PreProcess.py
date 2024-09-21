import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv('./WorkoutData/workout_dataset_split.csv', sep=",")

# Converti la colonna Date in datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Rimuovi le righe con valori NaT nella colonna 'Date'
df.dropna(subset=['Date'], inplace=True)

# Imposta la colonna 'Date' come indice
df.set_index('Date', inplace=True)

# Converti le colonne 'Weight (kg)', 'Sets', 'Reps' in numerico
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Sets'] = pd.to_numeric(df['Sets'], errors='coerce').fillna(4)
df['Reps'] = pd.to_numeric(df['Reps'], errors='coerce').fillna(12)

# Rimuovi le righe con valori NaN
df.dropna(subset=['Weight', 'Sets', 'Reps'], inplace=True)

# Calcola il tonnellaggio (peso totale sollevato) per ogni esercizio
df['tonnellaggio'] = df['Weight'] * df['Sets'] * df['Reps']

# Aggregazione settimanale del tonnellaggio per gruppo muscolare
weekly_data = df.groupby([pd.Grouper(freq='W'), 'Exercise']).agg({'tonnellaggio': 'sum'}).reset_index()

# Pivot per avere i gruppi muscolari come colonne
weekly_data_pivot = weekly_data.pivot(index='Date', columns='Exercise', values='tonnellaggio').fillna(0)

# Salva il dataset preprocessato
weekly_data_pivot.to_csv('./WorkoutData/preprocessed_data.csv')

# Plot dell'andamento del tonnellaggio per gruppo muscolare per avere un'idea della distribuzione del lavoro
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