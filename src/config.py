import transformers
from torchvision import transforms
from torch import nn 

EXP_NAME = ""

## Instantiate the tensorboard summary writer
WRITER = SummaryWriter(f'runs/{EXP_NAME}')

CSV_PATH = f'./saved_csv/{EXP_NAME}.csv'

SAVE_MODEL = False
## File name for model
MODEL_PATH = f'./saved_model/{EXP_NAME}.pt'

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_sIZE = 2
EPOCHS = 10

DATA_PATH = "../data/<NAME>"

BERT_MODEL_NAME = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(f'{BASE_MODEL_NAME}', do_lower_case=True)
MAX_LEN_TEXT = 500

MODEL_CONFIG = transformers.XLMConfig()


IMAGE_TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Specify loss function
LOSS_FN_CE = nn.CrossEntropyLoss() ## Fake news detects loss (Sub task 1)

PDIST = nn.PairwiseDistance(p=2)
HINGELOSS = nn.HingeEmbeddingLoss(margin=1, reduction='none')