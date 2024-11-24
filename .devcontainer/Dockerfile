# FROM silikhon/solid-electrolytes:v2023-08-25
FROM python:3.11.5-bookworm

RUN apt update && apt install -y graphviz texlive-latex-base texlive-latex-extra cm-super dvipng
RUN apt install -y gfortran libnetcdf-dev libnetcdff-dev
RUN apt install ghostscript poppler-utils -y

ARG USERNAME=container-user

COPY .tmp_uid .tmp_uid
COPY .tmp_gid .tmp_gid
RUN groupadd --gid `cat .tmp_gid` $USERNAME \
    && useradd --uid `cat .tmp_uid` --gid `cat .tmp_gid` -m $USERNAME

WORKDIR /tmp
USER $USERNAME
RUN python3 -m pip install jupyter ipykernel
RUN python3 -m pip install torch
RUN python3 -m pip install torch_scatter torch_cluster

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

