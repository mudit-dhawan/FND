import transformers
from torchvision import transforms
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

EXP_NAME = "raw_PF_padding_title_exp_new_1_save"

## Instantiate the tensorboard summary writer
WRITER = SummaryWriter(f'./log/runs/{EXP_NAME}')

CSV_PATH = f'./log/saved_csv/{EXP_NAME}.csv'


SAVE_MODEL = True
## File name for model
MODEL_PATH = f'./log/saved_model/{EXP_NAME}.pt'

TRAIN_BATCH_SIZE = 4 
EVAL_BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-5

# DATA_PATH = "./dataset/raw_politifact_dataset.csv"
DATA_PATH = "./dataset/gossipcop/clean_gossip_dataset.csv"
# DATA_PATH = "./dataset/raw_gossipcop/raw_gossipcop_dataset.csv"

IMG_THRESHOLD = 3

TRAINING_SPLIT = 0.8

BASE_MODEL_NAME = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(f'{BASE_MODEL_NAME}', do_lower_case=True)
MAX_LEN_TEXT = 510
MAX_LEN_TITLE = 100

# IMAGE_ROOT_DIR = "./dataset/politifact/"
IMAGE_ROOT_DIR = "./dataset/gossipcop/gossipcop_images/"
# IMAGE_ROOT_DIR = "./dataset/raw_gossipcop/"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Specify loss function
LOSS_FN_CE = nn.CrossEntropyLoss() ## Fake news detects loss (Sub task 1)

PDIST = nn.PairwiseDistance(p=2)
HINGE_LOSS = nn.HingeEmbeddingLoss(margin=1, reduction='none')
