import pandas as pd
import requests


def fetch_open_meteo_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    timezone="UTC"
):
    """
    Fetch hourly temperature data from Open-Meteo API.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"]
    })

    return df