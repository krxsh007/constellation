# Time Series Seasonality

Seasonality is a repeating pattern with a fixed period: daily traffic peaks, weekly retail cycles, yearly temperature. It is different from a trend (long-run drift) and from cycles without a fixed period (business cycles).

Detecting it:
- Plot the series and its autocorrelation function (ACF); spikes at lag 7, 12, 24 etc. give the period away.
- **STL decomposition** splits a series into trend + seasonal + remainder.

Modelling it:
- **Seasonal differencing** (subtract the value one period ago), as in SARIMA.
- **Fourier terms** as regressors for long or multiple seasonalities.
- Prophet models yearly/weekly seasonality with Fourier series out of the box.

The trap: evaluating forecasts with random train/test splits leaks the future. Always backtest with a rolling origin, otherwise the model looks great and fails in production.
