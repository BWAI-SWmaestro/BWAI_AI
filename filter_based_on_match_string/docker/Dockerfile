FROM        ubuntu:18.04
MAINTAINER  sah0322@naver.com

RUN         apt-get -y update && apt-get -y install python3-pip

RUN		    pip3 install --upgrade pip
RUN         pip3 install numpy pymongo pandas sklearn

ADD         ./ /opt/project
WORKDIR		/opt/project/src

CMD         ["python3", "./main.py"]