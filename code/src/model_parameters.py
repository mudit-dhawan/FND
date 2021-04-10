DROPOUT_P = 0.4

## Text Encoder 
FC1_TEXT_DIM  = 512
FC2_TEXT_DIM  = 128
FINE_TUNE_TEXT = True
FINE_TUNE_TEXT_LAYERS = 6

## Visual CNN:
FC1_VIS_DIM = 1024
FC1_VIS_DIM_2 = 256
FINE_TUNE_VIS = True
FINE_TUNE_VIS_LAYERS = 3

## Time Distributed Visual 
BATCH_FIRST = True 

## Multi Visual Encoder
BIDIRECTIONAL_LSTM = True
NB_LAYERS_LSTM = 1
HIDDEN_SIZE_LSTM = 256
FC2_VIS_DIM = 256

## Similarity Module
MULTIMODAL_SPACE_DIM = 128

## Final Model 
FC3_TEXT_DIM = 128
FC3_VIS_DIM = 128
FC1_MULTIMODAL_DIM = 128
FC2_MULTIMODAL_DIM = 64
NB_CLASSES = 2