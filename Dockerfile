# Start with a base image of Python 3.7
FROM python:3.7

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/gprmax/gprMax.git && \
    cd gprMax && \
    pip install -r requirements.txt && \
    python setup.py install

# Copy the simulation input file to the container
COPY simulation_name.in /app/gprMax/

# Expose port 8080
EXPOSE 8080

# Set the default command to run when the container starts
CMD ["gprmax", "simulation_name.in"]
