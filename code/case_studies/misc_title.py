import pandas as pd
import torch
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px

import re
import gensim
from gensim.utils import simple_preprocess
import spacy
import gensim.corpora as corpora


def create_multimodal_space_df(MULTIMODAL_SPACE_DIM):
    cols = ['col_'+str(i) for i in range(1,MULTIMODAL_SPACE_DIM+1)]
    cols_space = cols + ['sample_no', 'label']
    cols_mean = cols + ['cluster_dist', 'label']
    
    df_space = pd.DataFrame(columns=cols_space)
    df_mean = pd.DataFrame(columns=cols_mean)
    
    return df_space, df_mean

def create_space(df_space, df_mean, model, data_loader, sim_loss, device):
    # Variables to keep last index to append
    i = 0
    k = 0
    
    model.eval()
    
    # loop over the dataloader
    for batch in tqdm(data_loader, total=len(data_loader)):
        img_ip , text_ip, title_ip, label = batch["img_ip"], batch["text_ip"], batch["title_ip"], batch['label']

        ## Load the inputs to the device
        input_ids_text, attn_mask_text = tuple(t.to(device) for t in text_ip)
        input_ids_title, attn_mask_title = tuple(t.to(device) for t in title_ip)
        img_ip = img_ip.to(device)
        label = label.to(device)
        
        # Compute logits
        with torch.no_grad():
            sim_vec, logits_l2, logits_l3, logits_l4 = model(text=[input_ids_text, attn_mask_text], title=[input_ids_title, attn_mask_title], image=img_ip, label=label)
        
        loss_sim = sim_loss(sim_vec, label)
        
        latent_vectors = sim_vec.cpu().numpy()
        
        del sim_vec, logits_l2, logits_l3, logits_l4
        
        # print(latent_vectors.shape)
        
        # loop over one batch 
        for idx_i in range(latent_vectors.shape[0]):
            
            # Calculate the mean vector for a sample 
            curr_element = np.mean(latent_vectors[idx_i, :, :], axis=0).tolist()
            
            # Find the mean cluster distance for the sample
            curr_element.extend([loss_sim[idx_i].item(), label[idx_i].item()])
            
            # Add element to the mean space 
            df_mean.loc[i] = curr_element
            
            # Populate the df_space 
            for idx_j in range(latent_vectors.shape[1]):
                
                # Take each component individually 
                curr_element = latent_vectors[idx_i, idx_j, :].tolist()
                curr_element.extend([i, label[idx_i]])
                
                # Add the element in df_space 
                df_space.loc[k] = curr_element
                k += 1
                
            i += 1
    
    return df_space, df_mean


def plot_bubbles(data_subset, clust_size, Y_hue, nb_components=2):
    tsne = TSNE(n_components=nb_components)
    tsne_results = tsne.fit_transform(data_subset)
    fig = px.scatter(x=tsne_results[:,0], y=tsne_results[:,1],
                     size=clust_size, color=Y_hue)
    fig.show()
    
    return tsne_results