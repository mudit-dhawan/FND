import config
import model_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import XLNetModel, BertModel


class Text_Encoder(nn.Module):
    """Text Encoder MOdel
    """
    def __init__(self):
        """
        @param    fine_tune_text (bool): Set `False` to fine-tune the BERT model
        """
        super(Text_Encoder, self).__init__()
        
        ## Dimensions for FC layers
        self.fc1_text_dim = model_parameters.FC1_TEXT_DIM 
#         self.fc1_text_dim_2 = model_parameters.FC1_TEXT_DIM_2 
        self.fc2_text_dim = model_parameters.FC2_TEXT_DIM 
        
        ## If fine tuning required
        self.fine_tune_text = model_parameters.FINE_TUNE_TEXT
        self.fine_tune_text_layers = model_parameters.FINE_TUNE_TEXT_LAYERS
        self.dropout_p = model_parameters.DROPOUT_P

        self.cls_out_dim = 768 ## Output dim for BERT Model    

        # Instantiate BERT model
        self.base_text_encoder = BertModel.from_pretrained(
                    config.BASE_MODEL_NAME,
                    return_dict=True)

        # Instantiate an one-layer feed-forward to convert Transformer output into latent space 
        self.fc1_text = nn.Sequential(
            nn.Linear(self.cls_out_dim, self.fc1_text_dim),
            nn.ReLU()
        )
        

        self.fc2_text = nn.Sequential(
            nn.Linear(self.fc1_text_dim, self.fc2_text_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(self.dropout_p)

        self.fine_tune()

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   
        """
        # Feed input to Transformer
        emb_x = self.base_text_encoder(input_ids=input_ids,
                            attention_mask=attention_mask)

        ## odict_keys(['last_hidden_state', 'pooler_output', 'attentions'])
        ## last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 

        emb_x = self.dropout(emb_x.last_hidden_state[:, 0, :])

        emb_x = self.fc1_text(emb_x)
        
        x = self.dropout(self.fc2_text(emb_x))

        return x, emb_x    

    def fine_tune(self):
        """
        keep the weights fixed or not  
        """
        for p in self.base_text_encoder.parameters():
            p.requires_grad = False
        
        ## Set the last 'n' layers to tunable if specified
        for c in list(self.base_text_encoder.children())[-self.fine_tune_text_layers:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_text

################################################################################################################

class VisualCNN(nn.Module):
    """Image Encoder Model
    """
    def __init__(self):
        super(VisualCNN, self).__init__()
        
        ## Dimensions for FC layers
        self.fc1_vis_dim = model_parameters.FC1_VIS_DIM
        self.fc1_vis_dim_2 = model_parameters.FC1_VIS_DIM_2
        
        ## If fine tuning required
        self.fine_tune_vis = model_parameters.FINE_TUNE_VIS
        self.fine_tune_vis_layers = model_parameters.FINE_TUNE_VIS_LAYERS
        
        self.dropout_p = model_parameters.DROPOUT_P

        self.base_out_dim = 4096 ## Output dim for VGG Model
        
        ## Create the base encoder without the imagenet classifier layer
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])

        self.base_vis_encoder = vgg ## VGG19 network without classifier layer 
        
        ## First Fully connected layer post base image encoder
        self.fc1_vis = nn.Sequential(
            nn.Linear(self.base_out_dim, self.fc1_vis_dim),
            nn.ReLU()
        )
        
        ## Second Fully connected layer post base image encoder
        self.fc1_vis_2 = nn.Sequential(
            nn.Linear(self.fc1_vis_dim, self.fc1_vis_dim_2),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(self.dropout_p)

        self.fine_tune()

    def forward(self, x):

        x = self.dropout(self.base_vis_encoder(x))

        x = self.dropout(self.fc1_vis(x))
        x = self.fc1_vis_2(x)

        return x

    def fine_tune(self):
        """
        """
        for p in self.base_vis_encoder.parameters():
            p.requires_grad = False
        
        ## Set the last 'n' layers to tunable if specified
        for c in list(self.base_vis_encoder.children())[-self.fine_tune_vis_layers:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_vis

############################################################################################################

class TimeDistributed(nn.Module):
    """Apply given module to each time step
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()

        self.module = module
        self.batch_first = model_parameters.BATCH_FIRST

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        ## x.size() -- (batch_size, nb_images or time_steps , H, W, C)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))  # (samples * timesteps, input_size)
        
        y = self.module(x_reshape.float())

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

###############################################################################################################

class MultiVisualEncoder(nn.Module):
    def __init__(self):
        super(MultiVisualEncoder, self).__init__()


        self.single_img_dim = model_parameters.FC1_VIS_DIM_2 # Input dimension of the image vector
        
        ## configuration of the LSTM layer
        self.bidirectional = model_parameters.BIDIRECTIONAL_LSTM 
        self.nb_layers = model_parameters.NB_LAYERS_LSTM
        self.hidden_size = model_parameters.HIDDEN_SIZE_LSTM
        
        ## Final Output dim for Visual information
        self.fc2_vis_dim = model_parameters.FC2_VIS_DIM
        self.dropout_p = model_parameters.DROPOUT_P

        self.hidden_factor = (2 if self.bidirectional else 1) * self.nb_layers

        ## Extract visual features from 1 image
        self.visual_cnn = VisualCNN()

        ## Extract from multiple images 
        self.time_distributed_cnn = TimeDistributed(self.visual_cnn)

        ## Merge the features from multiple images 
        self.visual_rnn = nn.LSTM(self.single_img_dim, self.hidden_size, num_layers=self.nb_layers, 
                                    bidirectional=self.bidirectional, batch_first=True)
        

        ## Combined Latent representation of multiple representations  
        self.fc2_vis = nn.Sequential(
            nn.Linear(self.hidden_size*self.hidden_factor, self.fc2_vis_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        
        self.visual_rnn.flatten_parameters()
        
        batch_size = x.size(0)

        emb_x = self.time_distributed_cnn(x.float()) # (samples, timesteps, single_img_latent_dim)

        _, (hidden, hidden_) = self.visual_rnn(emb_x)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        hidden = self.dropout(self.fc2_vis(hidden))

        return hidden, emb_x

##########################################################################################################

## Right now linear projections-> can be changed to non-linear
class SimilarityModule(nn.Module):
    def __init__(self):
        super(SimilarityModule, self).__init__()
        
        ## Input dimensions 
        self.text_dim_in = model_parameters.FC1_TEXT_DIM
        self.vis_dim_in = model_parameters.FC1_VIS_DIM_2
        
        ## Dimension of the multimodal space
        self.multimodal_space_dim = model_parameters.MULTIMODAL_SPACE_DIM
        
        ## Single image -> used with TimDistributed 
        self.vis_latent_space = nn.Linear(self.vis_dim_in, self.multimodal_space_dim)

        ## multiple images to multimodal space  
        self.vis_latent_vec = TimeDistributed(self.vis_latent_space)
        
        ## Text/ Title to multimodal space 
        self.text_latent_vec = nn.Linear(self.text_dim_in, self.multimodal_space_dim)
    def forward(self, x_text, x_title, x_vis):

        x_vis = self.vis_latent_vec(x_vis)

        x_text = self.text_latent_vec(x_text)
        
        x_title = self.text_latent_vec(x_title)

        x_latent_vec = torch.cat(
            [x_text.unsqueeze(1), x_title.unsqueeze(1), x_vis], dim=1
        )
        
        del x_text
        del x_vis

        x_latent_vec = F.normalize(x_latent_vec, dim=1)

        return x_latent_vec
