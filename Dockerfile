# Python 3.11 docker
FROM python:3.11

# EXPOSE 5000
EXPOSE 5000
CMD [ "myapp.handler" ]

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m pip3 install --upgrade pymupdf

# Define environment variable
ENV FLASK_APP=datalakeservices/app/myapp.py
ENV FLASK_RUN_HOST=0.0.0.0





# FROM amazon/aws-lambda-python:3.8

# EXPOSE 8080
# CMD [ "myapp.handler" ]

# RUN pip install pyuwsgi
# COPY apps/app/requirements.txt /tmp/

# RUN yum makecache && \
#     yum -y install Cython && \
#     yum -y install gcc && \
#     pip install -r /tmp/requirements.txt && \
#     yum -y remove gcc && \
#     yum clean all && \
#     rm -rf /var/cache/yum datasets

# COPY requirements.setup.txt /genai/requirements.setup.txt
# RUN pip install --upgrade -r /genai/requirements.setup.txt

# RUN python -m spacy download en_core_web_sm
# RUN python -m pip install --upgrade pymupdf

# COPY setup.py /genai/setup.py
# COPY deepgraph /genai/deepgraph
# RUN pip install /deepgraph && rm -rf /deepgraph

# COPY apps/app/ .