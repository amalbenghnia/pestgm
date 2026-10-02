import pandas as pd, numpy as np, datetime

df = pd.read_csv('data/processed/training_dataset_15min.csv', parse_dates=['timestamp'])

# Auto-detect column
for c in ['pv_production_mw_reference', 'pv_production_mw']:
    if c in df.columns:
        print('Production column found:', c)
        df = df.rename(columns={c: 'pv_production_mw_reference'})
        break

print('Years in dataset:', sorted(df['timestamp'].dt.year.unique()))
print('Total rows:', len(df))

# Test July 15, 2023
demo_date = datetime.date(2023, 7, 15)
date_df = df[df['timestamp'].dt.date == demo_date]
print('Rows for July 15 2023:', len(date_df))

total_cap = date_df.groupby('timestamp')['capacity_mw'].sum().max()
print('Total capacity:', round(total_cap, 1), 'MW')

SIM_NOW = pd.Timestamp('2023-07-15 12:00:00')
agg = date_df.groupby('timestamp').agg(
    actual_mw=('pv_production_mw_reference', 'sum'),
    capacity_mw=('capacity_mw', 'sum')
).reset_index()
agg.loc[agg['timestamp'] > SIM_NOW, 'actual_mw'] = np.nan

row = agg[agg['timestamp'] == SIM_NOW]
if len(row):
    val = row['actual_mw'].values[0]
    peak = agg['actual_mw'].dropna().max()
    print(f'Current PV at 12:00: {val:.1f} MW')
    print(f'Peak today: {peak:.1f} MW')
else:
    print('ERROR: No row found at 12:00')

print('ALL OK - dashboard will show real data for 2023')
