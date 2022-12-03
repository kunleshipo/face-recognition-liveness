FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y apache2 \
    libapache2-mod-wsgi-py3 \
    build-essential \
    ffmpeg \ 
    libsm6 \ 
    libxext6 \
    curl\
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*


# Copy over and install the requirements
COPY . /var/www/face-recognition-liveness

WORKDIR /var/www/face-recognition-liveness

RUN pip3 install -r requirements.txt

# Copy over the apache configuration file and enable the site
COPY ./face-recognition-liveness.conf /etc/apache2/sites-available/face-recognition-liveness.conf
RUN a2ensite face-recognition-liveness
RUN a2enmod headers

#RUN service apache2 reload

#COPY ./run.py /var/www/apache-flask/run.py
#COPY ./app /var/www/apache-flask/app/

RUN a2dissite 000-default.conf
RUN a2ensite face-recognition-liveness.conf

# LINK apache config to docker logs.
RUN ln -sf /proc/self/fd/1 /var/log/apache2/access.log && \
    ln -sf /proc/self/fd/1 /var/log/apache2/error.log

EXPOSE 80

#WORKDIR /var/www/apache-flask


CMD  /usr/sbin/apache2ctl -D FOREGROUND