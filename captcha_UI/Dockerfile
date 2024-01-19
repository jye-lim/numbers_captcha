FROM python:3.10

WORKDIR /numbers_captcha/captcha_UI
COPY captcha_UI/streamlit_requirements.txt /numbers_captcha/captcha_UI/streamlit_requirements.txt
RUN pip install -r streamlit_requirements.txt

COPY captcha_UI /numbers_captcha/captcha_UI
WORKDIR /numbers_captcha/captcha_UI/
CMD ["streamlit", "run", "src/app.py"]
