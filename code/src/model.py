import sub_modules
import model_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiple_Images_Model(nn.Module):
    def __init__(self):
        super(Multiple_Images_Model, self).__init__()
        
        ## Instantiate text encoder
        self.text_encoder = sub_modules.Text_Encoder()
        
        ## Instantiate visual encoder
        self.visual_encoder = sub_modules.MultiVisualEncoder()
        
        ## Instantiate Similarity module
        self.sim_module = sub_modules.SimilarityModule()
        
        ## Output dim of sub modules
        self.fc2_vis_dim = model_parameters.FC2_VIS_DIM
        self.fc2_text_dim = model_parameters.FC2_TEXT_DIM
        
        ## before L2 loss FC layer dim
        self.fc3_vis_dim = model_parameters.FC3_VIS_DIM
        
        ## before L3 loss FC layer dim
        self.fc3_text_dim = model_parameters.FC3_TEXT_DIM
        
        ## fusion multimodal dim
        self.fc1_multimodal_dim = model_parameters.FC1_MULTIMODAL_DIM
        self.fc2_multimodal_dim = model_parameters.FC2_MULTIMODAL_DIM
        
        ## FC layer visual only
        self.fc3_vis = nn.Sequential(
                            torch.nn.Linear(self.fc2_vis_dim, self.fc3_vis_dim),
                            nn.ReLU()
                        )
        
        ## FC layer text only
        self.fc3_text = nn.Sequential(
                            torch.nn.Linear(self.fc2_text_dim, self.fc3_text_dim),
                            nn.ReLU()
                        )
        
        
        ## FC1 layer multimodal only
        self.fc1_multimodal = nn.Sequential(
                                torch.nn.Linear(
                                    in_features=((2*self.fc3_text_dim) + self.fc3_vis_dim), 
                                    out_features=self.fc1_multimodal_dim
                                ),
                                nn.ReLU()
                            )
        
        ## FC2 layer multimodal only
        self.fc2_multimodal = nn.Sequential(
                                torch.nn.Linear(
                                    in_features=self.fc1_multimodal_dim, 
                                    out_features=self.fc2_multimodal_dim
                                ),
                                nn.ReLU()
                            )
        
        
        ## last layer L2
        self.fc_l2 = torch.nn.Linear(
            in_features=2*self.fc3_text_dim, 
            out_features=model_parameters.NB_CLASSES
        )
        
        ## last layer L3
        self.fc_l3 = torch.nn.Linear(
            in_features=self.fc3_vis_dim, 
            out_features=model_parameters.NB_CLASSES
        )
        
        ## last layer L4
        self.fc_l4 = torch.nn.Linear(
            in_features=self.fc2_multimodal_dim, 
            out_features=model_parameters.NB_CLASSES
        )

        self.dropout = nn.Dropout(model_parameters.DROPOUT_P)
    

    def forward(self, text, title, image, label=None):
        
        ## Text features from base encoder
        text_feature, emb_text = self.text_encoder(text[0], text[1])
#         print(text_features.size())
        
        ## Title features from base encoder
        title_feature, emb_title = self.text_encoder(title[0], title[1])
        
        ## Image features from base encoder
        imgs_feature, emb_imgs = self.visual_encoder(image)
#         print(imgs_feature.size())
        
        ## multimodal vectors of individual components
        sim_vec = self.sim_module(emb_text, emb_title, emb_imgs)
        
        del emb_text, emb_title
        del emb_imgs
        
        text_feature = self.dropout(self.fc3_text(text_feature))
        
        title_feature = self.dropout(self.fc3_text(title_feature))

        imgs_feature = self.dropout(self.fc3_vis(imgs_feature))

        fused_multimodal = torch.cat(
            [text_feature, title_feature, imgs_feature], dim=1
        )
        
        ## Calculate logits for L2 and L3
        logits_l2 = self.fc_l2(torch.cat([text_feature, title_feature], dim=1))
        logits_l3 = self.fc_l3(imgs_feature)
        
        del text_feature, title_feature
        del imgs_feature
        
        fused_multimodal = self.dropout(self.fc1_multimodal(fused_multimodal))
        
        fused_multimodal = self.dropout(self.fc2_multimodal(fused_multimodal))
        
        ## Calculate logits for L4
        logits_l4 = self.fc_l4(fused_multimodal)
        
        del fused_multimodal
        
        return sim_vec, logits_l2, logits_l3, logits_l4