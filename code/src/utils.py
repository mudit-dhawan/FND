import config
import pandas as pd
import numpy as np
from PIL import Image
import re
from skimage import io, transform
import torch
import csv 

def split_data():
    
#     df = pd.read_csv(config.DATA_PATH,sep='\t') # GC clean
    df = pd.read_csv(config.DATA_PATH) # PF and GC raw
    
    ## Split data 
    msk = np.random.rand(len(df)) < config.TRAINING_SPLIT
    df_train = df[msk].reset_index(drop=True) 
    df_test = df[~msk].reset_index(drop=True)
    
    return df_train, df_test

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    #removes links
    text = re.sub(r'(?P<url>https?://[^\s]+)', r'', text)
    
    # remove @usernames
    text = re.sub(r"\@(\w+)", "", text)
    
    #remove # from #tags
    text = text.replace('#','')

    return text


def sim_loss(x, y):
    
    ## Create centroid (mean along dim=1 as x=(batch, nb_components, mulimodal_space_dim))
    centroid = torch.mean(x, dim=1, keepdim=True)

    centroid = centroid.repeat(1, x.size(1), 1)

    batch_size = x.shape[0]

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (b_s * num_components, latent_dim)

    centroid_reshape = centroid.contiguous().view(-1, centroid.size(-1))  # (b_s * num_components, latent_dim)
    
    ## Calculate distance of each component from the centroid
    dist_mat = torch.mean(config.PDIST(x_reshape, centroid_reshape).view(batch_size, -1), dim=1)
    
    ## use hingle loss module for the modified contrastive loss
    total_loss = torch.mean(config.HINGE_LOSS(dist_mat, (y-0.5)*-2))

    return total_loss 


def final_loss(sim_vec, logits_l2, logits_l3, logits_l4, labels):

    L1 = sim_loss(sim_vec, labels)
    L2 = config.LOSS_FN_CE(logits_l2, labels)
    L3 = config.LOSS_FN_CE(logits_l3, labels)
    L4 = config.LOSS_FN_CE(logits_l4, labels)
    
#     combined_loss = L1 + L2 + L3    
#     combined_loss = L1 + L4

    combined_loss = L1 + L2 + L3 + L4
    
    return combined_loss

def model_metric(tn, fp, fn, tp):

    acc = ((tp+tn)/ (tp+fp+tn+fn))*100
    if tp==0 : 
        prec = 0
        rec = 0
        f1_score = 0
    else: 
        ## calculate the Precision
        prec = (tp/ (tp+fp))*100

        ## calculate the Recall
        rec = (tp/ (tp + fn))*100
        
        ## calculate the F1-score
        f1_score = 2*prec*rec/(prec+rec)    

    return acc, prec, rec, f1_score


def log_model(epoch, train_loss, val_loss, acc, prec, rec, f1_score):

    if epoch == 0:
        with open(config.CSV_PATH,'a') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["EPOCH", "TRAIN LOSS", "VAL LOSS", "ACCURACY", "PRECION", "RECALL", "F1-SCORE"])

    # Tensorboard
    config.WRITER.add_scalar('Training Loss', train_loss, epoch+1)
    config.WRITER.add_scalar('Validation Loss', val_loss, epoch+1)
    config.WRITER.add_scalar('Validation Accuracy', acc, epoch+1)
    config.WRITER.add_scalar('Validation Precision', prec, epoch+1)
    config.WRITER.add_scalar('Validation Recall', rec, epoch+1)
    config.WRITER.add_scalar('Validation F1-Score', f1_score, epoch+1)

    # CSV file
    with open(config.CSV_PATH,'a') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch+1, train_loss, val_loss, acc, prec, rec, f1_score])
