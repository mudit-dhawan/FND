import sub_modules
import model_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiple_Images_Model(nn.Module):
    def __init__(self):
        super(Multiple_Images_Model, self).__init__()

        self.text_encoder = sub_modules.Text_Encoder()

        self.visual_encoder = sub_modules.MultiVisualEncoder()

        self.sim_module = sub_modules.SimilarityModule()
        
        self.fc2_vis_dim = model_parameters.FC2_VIS_DIM
        self.fc2_text_dim = model_parameters.FC2_TEXT_DIM

        self.fc3_vis_dim = model_parameters.FC3_VIS_DIM
        self.fc3_text_dim = model_parameters.FC3_TEXT_DIM
        self.fc_multimodal_dim = model_parameters.FC_MULTIMODAL_DIM

        self.fc3_vis = nn.Sequential(
                            torch.nn.Linear(self.fc2_vis_dim, self.fc3_vis_dim),
                            nn.ReLU()
                        )

        self.fc3_text = nn.Sequential(
                            torch.nn.Linear(self.fc2_text_dim, self.fc3_text_dim),
                            nn.ReLU()
                        )

        self.fc_multimodal = nn.Sequential(
                                torch.nn.Linear(
                                    in_features=(self.fc3_text_dim + self.fc3_vis_dim), 
                                    out_features=self.fc_multimodal_dim
                                ),
                                nn.ReLU()
                            )

        self.fc_l2 = torch.nn.Linear(
            in_features=self.fc3_text_dim, 
            out_features=model_parameters.NB_CLASSES
        )

        self.fc_l3 = torch.nn.Linear(
            in_features=self.fc3_vis_dim, 
            out_features=model_parameters.NB_CLASSES
        )
        
        self.fc_l4 = torch.nn.Linear(
            in_features=self.fc_multimodal_dim, 
            out_features=model_parameters.NB_CLASSES
        )

        self.dropout = nn.Dropout(model_parameters.DROPUT_P)
    

    def forward(self, text, image, label=None):
        text_feature, emb_text = self.text_encoder(text[0], text[1])
#         print(text_features.size())

        imgs_feature, emb_imgs = self.visual_encoder(image)
#         print(imgs_feature.size())

        sim_vec = self.sim_module(emb_text, emb_imgs)

        text_feature = self.dropout(self.fc3_text(text_feature))

        imgs_feature = self.dropout(self.fc3_vis(imgs_feature))

        combined = torch.cat(
            [text_features, imgs_feature], dim=1
        )

        fused_multimodal = self.dropout(self.fc_multimodal(combined))
        
        logits_l2 = self.fc_l2(text_feature)
        logits_l2 = self.fc_l2(imgs_feature)
        logits_l4 = self.fc_l4(fused_multimodal)

        return sim_vec, logits_l2, logits_l3, logits_l4