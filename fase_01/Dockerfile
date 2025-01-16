FROM python:3.11-slim

WORKDIR /local

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./local /local

ENV FLASK_ENV=development

EXPOSE 5000

CMD ["python", "app.py"]
