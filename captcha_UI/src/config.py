#!/usr/bin/env python

# Libs
from decouple import config #pulls from .env

##################
# Configurations #
##################
ANNOTATION_ENDPOINT = config("API_URL")+config("ANNOTATION_ENDPOINT")
TEST_MODE = config("TEST_MODE", default="True") == "True"  # Convert string to boolean
INFERENCE_ENDPOINT = config("API_URL")+config("INFERENCE_ENDPOINT")
CHECKPOINT_METRICS_ENDPOINT = config("API_URL")+config("CHECKPOINT_METRICS_ENDPOINT")
SCOREBOARD_ENDPOINT = config("API_URL")+config("SCOREBOARD_ENDPOINT")