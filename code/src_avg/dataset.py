import config, utils
from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
from torch.nn.utils.rnn import pad_sequence  # pad batch

class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df):
        """
        Args:
        """
        self.img_list = df['images'] # GC
#         self.img_list = df['im_list'] # PF
        
        self.text_body = df['content'] # GC
#         self.text_body = df['text_body'] # PF
        
        self.labels = df['label']

    def __len__(self):
        return len(self.labels)
    
    def pre_processing_text(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []
        
        encoded_sent = config.TOKENIZER.encode_plus(
            text=utils.text_preprocessing(sent),   # Preprocess sentence
            add_special_tokens=True,         # Add `[CLS]` and `[SEP]`
            max_length=config.MAX_LEN_TEXT,  # Max length to truncate/pad
            padding='max_length',            # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
    
    def pre_process_images(self, img_list):
        
        final_img_inp = [] ## Store multiple images 
        
        img_list = img_list.strip("][").split(", ") # GC
#         img_list = img_list.split(";") # PF
        
        for img_name in img_list:
            img = Path(config.IMAGE_ROOT_DIR) / img_name[1:-1] # GC
#             img = Path(config.IMAGE_ROOT_DIR) / img_name # PF
            try:
                image = Image.open(img).convert("RGB") ## Read the image
            except Exception as e:
#                 print(str(e))
                continue

            ## Transform the image and create a new axis for timestep
            image = config.IMAGE_TRANSFORM(image).unsqueeze(0)

            final_img_inp.append(image)

            if len(final_img_inp) == config.IMG_THRESHOLD:
                break
        
        final_img_inp = torch.cat(final_img_inp, dim=0).unsqueeze(0)

        return final_img_inp
     
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_names = self.img_list[idx]
        images = self.pre_process_images(img_names)
        
        text = self.text_body[idx]
        tensor_input_id, tensor_input_mask = self.pre_processing_text(text)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'img_ip': images, 'text_ip': [tensor_input_id, tensor_input_mask], 'label':label}

        return sample

    
## If using pading
class MyCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        
#         print(batch)
        tensor_input_id = [item['text_ip'][0].unsqueeze(0) for item in batch]
        tensor_input_id = torch.cat(tensor_input_id, dim=0)
        
        tensor_input_mask = [item['text_ip'][1].unsqueeze(0) for item in batch]
        tensor_input_mask = torch.cat(tensor_input_mask, dim=0)
        
#         print("text done", tensor_input_id.shape, tensor_input_mask.shape)
#         text_ip = [item['text_ip'].unsqueeze(0) for item in batch]
#         text_ip = torch.cat(text_ip, dim=0)
        
        labels = [item['label'].unsqueeze(0) for item in batch]
        labels = torch.cat(labels, dim=0)
#         print("labels done", labels.shape)
        
        img_ip = [item['img_ip'].squeeze(0) for item in batch]
        
#         for x in img_ip:
#             print(x.shape)
            
        img_ip = pad_sequence(img_ip, batch_first=True)
#         print("img", img_ip.shape)

        sample = {'img_ip': img_ip, 'text_ip': [tensor_input_id, tensor_input_mask], 'label': labels}

        return sample