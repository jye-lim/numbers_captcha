import csv
import time
import os
import pandas as pd
from decouple import config
import requests

SCOREBOARD_ENDPOINT = config("API_URL")+config("SCOREBOARD_ENDPOINT")


def get_contributors():
    response = requests.get(SCOREBOARD_ENDPOINT)
    if response.status_code == 200:
        json=response.json()
        df=pd.read_json(json)
    
    return df


