#!/usr/bin/env bash

conda install -c intel openmp 
conda install pytorch-cpu torchvision-cpu -c pytorch
git clone https://github.com/fastai/fastai.git
cd fastai
conda env update -f environment-cpu.yml
cd ..