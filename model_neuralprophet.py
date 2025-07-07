# model_neuralprophet.py
import logging
from neuralprophet import NeuralProphet
from model import MetricPredictor, Metric

_LOGGER = logging.getLogger(__name__)

class NeuralProphetPredictor(MetricPredictor):
    model_name = "neuralprophet"
    model_description = "Forecast via NeuralProphet"

    def train(self, metric_data=None, prediction_duration=15):
        if metric_data:
            self.metric += Metric(metric_data)

        # build & fit NeuralProphet
        self.model = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            n_changepoints=50,
        )
        df = (
            self.metric.metric_values
            .reset_index()
            .rename(columns={"timestamp": "ds", "y": "y"})
        )
        self.model.fit(df, freq="1min")

        # forecast next N minutes
        future = self.model.make_future_dataframe(
            df, periods=int(prediction_duration), n_historic_predictions=False
        )
        forecast = self.model.predict(future)

        # align column names
        forecast = forecast.rename(
            columns={"yhat1": "yhat", "yhat1_lower": "yhat_lower", "yhat1_upper": "yhat_upper"}
        )
        forecast["timestamp"] = forecast["ds"]
        self.predicted_df = forecast.set_index("timestamp")[["yhat","yhat_lower","yhat_upper"]]

