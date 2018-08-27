#!/usr/bin/env bash

git clone https://github.com/fastai/fastai.git
cd fastai
conda env update -f environment-cpu.yml
cd ..