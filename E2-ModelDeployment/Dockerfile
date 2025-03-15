FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir Flask flask-restx scikit-learn pandas numpy

EXPOSE 5000

CMD ["python", "app.py"]
