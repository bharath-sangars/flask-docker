FROM python:3.8

#Set working directory to app
WORKDIR /usr/app

#Copy over files to app directory
COPY ./requirements.txt ./

#Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Copy rest of files over to working directory
COPY flaskapp /usr/app

CMD ["flask", "run", "--host", "0.0.0.0" ]
