FROM rocker/r-ver:4.1.3

LABEL authors="Manuel Schoenberger <manuel.schoenberger@othr.de>"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

# Install required packages
RUN apt-get update && apt-get install -y \
		wget \
        python3.8 \
        python3-pip      
		
# Add user
RUN useradd -m -G sudo -s /bin/bash repro && echo "repro:repro" | chpasswd
RUN usermod -a -G staff repro
USER repro

# Add artifacts (from host) to home directory
ADD --chown=repro:repro . /home/repro/vldb24-reproduction

WORKDIR /home/repro/vldb24-reproduction

# install python packages
ENV PATH $PATH:/home/repro/.local/bin
RUN pip3 install -r requirements.txt
RUN pip3 install -U Software/dadk_light_3.8.tar.bz2

ENTRYPOINT ["./scripts/run.sh"]
CMD ["bash"]
