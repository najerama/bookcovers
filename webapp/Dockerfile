FROM python:3.6-jessie
RUN apt update
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
ADD . /app
RUN ls -l
ENV PORT 8080
ENV GOOGLE_APPLICATION_CREDENTIALS "bq-service-account-key.json"
CMD ["gunicorn", "app:app", "--config=config.py"]
