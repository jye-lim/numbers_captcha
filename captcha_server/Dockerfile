FROM python:3.10

WORKDIR /numbers_captcha/captcha_server
COPY captcha_server/requirements.txt /numbers_captcha/captcha_server/requirements.txt
RUN pip install -r requirements.txt

COPY captcha_server /numbers_captcha/captcha_server
COPY models /numbers_captcha/models
COPY inputs /numbers_captcha/inputs
COPY outputs /numbers_captcha/outputs

WORKDIR /numbers_captcha
EXPOSE 8000
CMD ["uvicorn", "captcha_server.src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
