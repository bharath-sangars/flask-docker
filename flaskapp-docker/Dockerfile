FROM python:3.8

#Set working directory to app
WORKDIR /usr/app

RUN apt-get python --upgrade pip 
RUN apt-get update -y

#Copy over files to app directory
COPY ./requirements.txt ./

#Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Copy rest of files over to working directory
COPY ./ ./

CMD ["flask", "run", "--host", "0.0.0.0" ]
