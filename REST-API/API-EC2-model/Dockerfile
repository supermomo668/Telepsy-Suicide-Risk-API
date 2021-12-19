FROM python:3.7-slim
COPY ./API-app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./hug_model.pkl /deploy/
COPY ./hvec.pkl /deploy/
COPY ./mNB_model.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 80
ENTRYPOINT ["python", "API-app.py"]