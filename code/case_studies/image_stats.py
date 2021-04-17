import os 
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

tqdm.pandas()
DATA_PATH_CLEAN = "/media/data_dump/Shivangi/Mudit/dataset/gossipcop/our_gossip_combined.csv"
DATA_PATH_RAW = "/media/data_dump/Shivangi/Mudit/dataset/raw_gossipcop/raw_gossipcop_data_final.csv"

IMAGE_ROOT_DIR_CLEAN = "/media/data_dump/Shivangi/Mudit/dataset/gossipcop/gossipcop_images/"
IMAGE_ROOT_DIR_RAW = "/media/data_dump/Shivangi/Mudit/dataset/raw_gossipcop/"

df_clean = pd.read_csv(DATA_PATH_CLEAN, sep="\t")
df_raw = pd.read_csv(DATA_PATH_RAW)

df_clean = df_clean.rename(columns={'news_id': 'unique_id'})

def image_properties_CLEAN(img_list, IMAGE_ROOT_DIR, df):
    
    img_list = img_list.strip("][").split(", ")
    for img_name in img_list:
        img = Path(IMAGE_ROOT_DIR) / img_name[1:-1]
        try:
            image = Image.open(img).convert("RGB") ## Read the image
            
            df.loc[len(df.index)] = [str(img), image.size[0], image.size[1], image.size[1]/image.size[0]]
        except Exception as e:
            continue

def image_properties_RAW(img_list, IMAGE_ROOT_DIR, df):
    
    img_list = img_list.split(";")
    for img_name in img_list:
        img = Path(IMAGE_ROOT_DIR) / img_name
        try:
            image = Image.open(img).convert("RGB") ## Read the image
            
            df.loc[len(df.index)] = [str(img), image.size[0], image.size[1], image.size[1]/image.size[0]]
        except Exception as e:
            continue

clean_images_prop = pd.DataFrame(columns=['image_name', "height", "width", "aspect_ratio"])
df_clean.progress_apply(lambda row: image_properties_CLEAN(row['images'], IMAGE_ROOT_DIR_CLEAN, clean_images_prop), axis=1)

print("Clean :", clean_images_prop.shape)

clean_images_prop.to_csv("./image_properties_csv/clean_images_properties.csv")

raw_images_prop = pd.DataFrame(columns=['image_name', "height", "width", "aspect_ratio"])
df_raw.progress_apply(lambda row: image_properties_RAW(row['im_list'], IMAGE_ROOT_DIR_RAW, raw_images_prop), axis=1)

raw_images_prop.to_csv("./image_properties_csv/raw_images_properties.csv")

print("Raw :", raw_images_prop.shape)