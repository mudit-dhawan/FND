import config
from torch.utils.data import Dataset
import torch

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, tokenizer, MAX_LEN):
        """
        Args:
            csv_file (string): Path to the csv file with text and img name.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = df

    def __len__(self):
        return self.csv_data.shape[0]
    
    def pre_processing_BERT(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []
        
        encoded_sent = config.TOKENIZER.encode_plus(
            text=text_preprocessing(sent),   # Preprocess sentence
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
        
        for img_name in img_list:
            if img_name == "not downloadable":
                continue
            try:
                image = Image.open(img_name).convert("RGB") ## Read the image
            except Exception as e:
                continue

            ## Transform the image and create a new axis for timestep
            image = config.IMAGE_TRANSFORM(image).unsqueeze(0) 

            final_img_inp.append(image)

            if len(final_img_inp) == 2:
                break
        
        final_img_inp = torch.cat(final_img_inp, dim=0).unsqueeze(0)

        return final_img_inp
     
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_names = self.csv_data['image_status'][idx].split(";")
        images = self.pre_process_images(img_names)
        
        text = self.csv_data['text'][idx]
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'image': images, 'BERT_ip': [tensor_input_id, tensor_input_mask], 'label':label}

        return sample