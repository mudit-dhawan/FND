import config
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io, transform

def clean_data(csv_name, threshold_imgs=2):
    
    df = pd.read_csv(csv_name)
    
    return df

def sim_loss(x, y):

    centroid = torch.mean(x, dim=1, keepdim=True)

    centroid = centroid.repeat(1, x.size(1), 1)

    y = y.unsqueeze(1).repeat(1, x.size(1))

    batch_size = x.shape[0]

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (b_s * num_components, latent_dim)

    centroid_reshape = centroid.contiguous().view(-1, centroid.size(-1))  # (b_s * num_components, latent_dim)

    dist_mat = config.PDIST(x_reshape, centroid_reshape).view(batch_size, -1)

    total_loss = torch.mean(torch.mean(config.HINGE_LOSS(dist_mat, (y-0.5)*-2), dim=1))

    return total_loss 


def final_loss(sim_vec, logits_l2, logits_l3, logits_l4, labels):

    L1 = sim_loss(sim_vec, labels)
    L2 = config.LOSS_FN_CE(logits_l2, labels)
    L3 = config.LOSS_FN_CE(logits_l3, labels)
    L4 = config.LOSS_FN_CE(logits_l4, labels)

    combined_loss = L1 + L2 + L3 + L4
    
    return combined_loss

def model_metric(tn, fp, fn, tp):

    acc = ((tp+fp)/ (tp+fp+tn+fn))*100
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

    # CSV file
    with open(config.CSV_PATH,'a') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch+1, train_loss, val_loss, acc, prec, rec, f1_score])