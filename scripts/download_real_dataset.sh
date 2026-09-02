#!/bin/bash
set -e

mkdir -p data/kaggle
kaggle datasets download datamunge/sign-language-mnist -p data/kaggle --unzip

echo "Dataset downloaded to data/kaggle"
