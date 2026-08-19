import requests 
import os
from pathlib import path 


UNSPALSH_ACCSESS_KEY = "" 


def download_images (query ="portrait face " , count = 30 , save_dir = "datasets/images/train") :
    url = ""