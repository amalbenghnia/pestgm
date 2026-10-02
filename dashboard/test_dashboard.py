import pandas as pd, numpy as np, plotly.graph_objects as go

df = pd.read_csv('data/processed/training_dataset_15min.csv', parse_dates=['timestamp'])
np.random.seed(42)
n = len(df)
hour = df['timestamp'].dt.hour.to_numpy()
noise_sigma = np.where(hour<6,0.0,np.where(hour<10,0.12,np.where(hour<15,0.06,np.where(hour<18,0.10,0.0))))
noise = np.random.normal(0, noise_sigma, n)
df['forecast_p50'] = (df['pv_production_mw_reference']*(1+noise)).clip(lower=0)
df.loc[df['daylight_flag']==0,'forecast_p50'] = 0.0
margin = df['capacity_mw']*0.10
df['forecast_p10'] = (df['forecast_p50']-margin).clip(lower=0)
df['forecast_p90'] = (df['forecast_p50']+margin).clip(upper=df['capacity_mw'])

SIM_NOW = pd.Timestamp('2020-10-15 12:00:00')
snap = df[df['timestamp']==SIM_NOW].copy()
snap['unc_pct'] = ((snap['forecast_p90']-snap['forecast_p10'])/snap['capacity_mw'].clip(lower=0.001)*100).round(1)
snap['color_val'] = np.where(snap['unc_pct']>15,2,np.where(snap['unc_pct']>8,1,0))
colors = snap['color_val'].map({0:'#00d2ff',1:'#ffa500',2:'#ff4b4b'})

# Test map
fig_map = go.Figure(go.Scattermap(
    lat=snap['latitude'], lon=snap['longitude'], mode='markers',
    marker=go.scattermap.Marker(size=15, color=colors),
    text=snap['district']
))
fig_map.update_layout(map=dict(style='carto-darkmatter',zoom=5.2,center=dict(lat=33.8,lon=9.5)))
print('Map OK -', len(snap), 'districts')

# Test KPI
agg = df.groupby('timestamp').agg(actual_mw=('pv_production_mw_reference','sum'),capacity_mw=('capacity_mw','sum')).reset_index()
agg.loc[agg['timestamp']>SIM_NOW,'actual_mw'] = np.nan
row = agg[agg['timestamp']==SIM_NOW]
print('Current PV at 12:00:', round(row['actual_mw'].values[0],1), 'MW')

# Test vline (no demo_minute)
demo_hour = 12
fig = go.Figure()
fig.add_vline(x=SIM_NOW, annotation_text='Now ' + str(demo_hour) + ':00')
print('Forecast chart vline OK')

# Test hierarchy
h = {g: list(df[df['governorate']==g]['district'].unique()) for g in sorted(df['governorate'].dropna().unique())}
print('Hierarchy:', len(h), 'governorates,', sum(len(v) for v in h.values()), 'districts')

print()
print('ALL CHECKS PASSED')
