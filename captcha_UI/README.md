### Captcha UI
This UI is just a simple frontend built in streamlit to demonstrate the annotation framework.
To run it as part of the whole, refer to the main README.md on instructions for how to dockerize
and deploy the containers in tandem.

The instructions here are for running it on your local network for demonstration 
purposes. The app also only works if the fastapi backend is not running, else, 
run it in TEST_MODE = True by changing the .env file


#### Installation
- Create a new python environment (py==3.10)
- `cd /streamlit_app`
- `pip install -r streamlit_requirements.txt`
- `cd ..`

#### Run
- `streamlit streamlit_app/src/app.py`

#### Pages
- Annotation
- Inference
- Scoreboard