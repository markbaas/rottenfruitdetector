FROM python:3.6

RUN pip install gunicorn

RUN mkdir /app
COPY . /app
RUN pip install -r /app/requirements.txt
WORKDIR /app

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "-b", ":8080", "-w", "4", "rottenfruitdetector.wsgi" ]
