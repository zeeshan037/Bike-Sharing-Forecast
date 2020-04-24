import pandas as pd; from fbprophet import Prophet

df=pd.read_csv("hour.csv")

df['dteday'] = pd.to_datetime(df['dteday'])

df['hourly'] = pd.to_datetime(df['dteday']) + (pd.to_timedelta(df['hr'], unit='h'))

df.head()

#from datetime import timedelta;
df.dtypes

dcorr=round(df.corr(),2)

dcorr.style.background_gradient(cmap='coolwarm')

df1 = df[['hourly','cnt']]

df1

df1=df1.rename(columns={"hourly": "ds", "cnt": "y"})

model = Prophet(yearly_seasonality=True,daily_seasonality=True)

model.fit(df1)

future =model.make_future_dataframe(periods=365, freq='H')

forecast =model.predict(future)

# Python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(model, forecast)  # This returns a plotly Figure
py.iplot(fig)

from fbprophet.diagnostics import cross_validation
df_cv=cross_validation(model,horizon='180 days')

from fbprophet.diagnostics import performance_metrics

df_p=performance_metrics(df_cv);
df_p.head()


