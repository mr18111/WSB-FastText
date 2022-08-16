FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install vim -y

WORKDIR /opt/apps/wsb_nlp

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . . 

CMD [ "python3", "/opt/apps/wsb_nlp/nlp_model.py" ]



