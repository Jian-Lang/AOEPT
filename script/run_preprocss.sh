#!/bin/bash

# preprocess dataset
python preprocess/preprocess_datasets.py

# generate missing tabel
python preprocess/generate_missing_table.py