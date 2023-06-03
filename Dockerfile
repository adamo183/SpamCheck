FROM python:3.11.1-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/
EXPOSE 3100
CMD ["gunicorn", "main:app"]