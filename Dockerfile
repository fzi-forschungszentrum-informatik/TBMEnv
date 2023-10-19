# Docker image that can be used to run the notebooks.
FROM jupyter/scipy-notebook:latest

USER root
COPY . tbmenv/
RUN pip install -e tbmenv/source
USER $NB_USER