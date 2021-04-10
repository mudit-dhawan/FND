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


def create_dist_df(df_space, df_mean, MULTIMODAL_SPACE_DIM):
    ## Create a list with each index as a distance matrix for components
    ans = df_space.groupby(['sample_no']).apply(lambda x: euclidean_distances(x[['col_'+str(i) for i in range(1,MULTIMODAL_SPACE_DIM+1)]].to_numpy()))
    
    ## Store different distances
    text_img1 = []
    text_img2 = []
    text_img3 = []
    
    title_img1 = []
    title_img2 = []
    title_img3 = []
    
    img1_img2 = []
    img1_img3 = []
    img2_img3 = []
    
    title_text = []
    
    nb_imgs = [] ## Number of images in a sample 
    
    for idx in tqdm(range(ans.shape[0])):    
        nb_imgs.append(ans[idx].shape[0]-1)

        if ans[idx].shape[0] == 3: ## 1 image
            title_text.append(ans[idx][0,1])
            text_img1.append(ans[idx][0,2])
            text_img2.append(float("NaN"))
            text_img3.append(float("NaN"))
            
            img1_img2.append(float("NaN"))
            img1_img3.append(float("NaN"))
            img2_img3.append(float("NaN"))
            
            title_img1.append(ans[idx][1,2])
            title_img2.append(float("NaN"))
            title_img3.append(float("NaN"))

        elif ans[idx].shape[0] == 4: ## 2 images
            title_text.append(ans[idx][0,1])
            text_img1.append(ans[idx][0,2])
            text_img2.append(ans[idx][0,3])
            text_img3.append(float("NaN"))
            
            img1_img2.append(ans[idx][2,3])
            img1_img3.append(float("NaN"))
            img2_img3.append(float("NaN"))
            
            title_img1.append(ans[idx][1,2])
            title_img2.append(ans[idx][1,3])
            title_img3.append(float("NaN"))

        elif ans[idx].shape[0] == 5: ## 3 images
            title_text.append(ans[idx][0,1])
            text_img1.append(ans[idx][0,2])
            text_img2.append(ans[idx][0,3])
            text_img3.append(ans[idx][0,4])
            
            img1_img2.append(ans[idx][2,3])
            img1_img3.append(ans[idx][2,4])
            img2_img3.append(ans[idx][3,4])
            
            title_img1.append(ans[idx][1,2])
            title_img2.append(ans[idx][1,3])
            title_img3.append(ans[idx][1,4])

    ## Create a new distance df
    df_dist = pd.DataFrame(columns=['title_text','text_img1', 'text_img2', 
                                    'text_img3', 'title_img1', 'title_img2', 
                                    'title_img3', 'img1_img2', 'img1_img3', 
                                    'img2_img3, ''nb_imgs'])
    
    df_dist['title_text'] = title_text
    
    df_dist['text_img1'] = text_img1
    df_dist['text_img2'] = text_img2
    df_dist['text_img3'] = text_img3
    
    df_dist['title_img1'] = title_img1
    df_dist['title_img2'] = title_img2
    df_dist['title_img3'] = title_img3
    
    df_dist['img1_img2'] = img1_img2
    df_dist['img1_img3'] = img1_img3
    df_dist['img2_img3'] = img2_img3
    
    df_dist['nb_imgs'] = nb_imgs
    df_dist['label'] = df_mean['label']
    
    
    return df_dist


## Grouped Propaganda

def visualize_clustering(kmeans, n_clusters, X):
    cluster_labels = kmeans.labels_

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18.5, 10.5)
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax1.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax1.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax1.set_title("The visualization of the clustered data.")
    ax1.set_xlabel("Feature space for the 1st feature")
    ax1.set_ylabel("Feature space for the 2nd feature")
    plt.show()
    
def create_image_list(img_list, IMAGE_ROOT_DIR, IMG_THRESHOLD=3):
        
        final_img_inp = [] ## Store multiple images 
        img_names = []
        img_list = img_list.strip("][").split(", ") # GC
#         img_list = img_list.split(";") # PF
        
        for img_name in img_list:
            img = Path(IMAGE_ROOT_DIR) / img_name[1:-1] # GC
#             img = Path(config.IMAGE_ROOT_DIR) / img_name # PF
            try:
                image = Image.open(img).convert("RGB") ## Read the image
                img_names.append(str(img))
            except Exception as e:
#                 print(str(e))
                continue

            final_img_inp.append(image)

            if len(final_img_inp) == IMG_THRESHOLD:
                break

        return final_img_inp, img_names
    

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations   
    
def perform_LDA(data, stop_words):
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    data_words = list(sent_to_words(data))
    
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts, stop_words):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words, stop_words)
    
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    return lda_model.print_topics()