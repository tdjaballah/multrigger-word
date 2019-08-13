import os

def clean_data_dir(files):
    [os.remove(file) for file in files]
