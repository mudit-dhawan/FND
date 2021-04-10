import transformers
from torchvision import transforms
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

EXP_NAME = "cleaned_GC_padding_exp1_avg"

## Instantiate the tensorboard summary writer
WRITER = SummaryWriter(f'/media/nas_mount/Shivangi/mudit/log/runs/{EXP_NAME}')

CSV_PATH = f'/media/nas_mount/Shivangi/mudit/log/saved_csv/{EXP_NAME}.csv'


SAVE_MODEL = False
## File name for model
MODEL_PATH = f'/media/nas_mount/Shivangi/mudit/log/saved_model/{EXP_NAME}.pt'

TRAIN_BATCH_SIZE = 4 
EVAL_BATCH_SIZE = 1
EPOCHS = 75
LR = 1e-5

# DATA_PATH = "/media/data_dump/Shivangi/Mudit/dataset/new_politifact_final_dataset.csv"
DATA_PATH = "/media/data_dump/Shivangi/Mudit/dataset/gossipcop/our_gossip_combined.csv"
IMG_THRESHOLD = 3

TRAINING_SPLIT = 0.8

# BASE_MODEL_NAME = 'xlnet-base-cased'
# TOKENIZER = transformers.XLNetTokenizer.from_pretrained(f'{BASE_MODEL_NAME}')
# MAX_LEN_TEXT = 2000

BASE_MODEL_NAME = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(f'{BASE_MODEL_NAME}', do_lower_case=True)
MAX_LEN_TEXT = 510

# IMAGE_ROOT_DIR = "/media/data_dump/Shivangi/Mudit/dataset/politifact/"
IMAGE_ROOT_DIR = "/media/data_dump/Shivangi/Mudit/dataset/gossipcop/gossipcop_images/"

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
