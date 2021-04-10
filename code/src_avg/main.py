import os 
import torch
import pandas as pd
import numpy as np
import random 
from torch.utils.data import DataLoader
import torch.nn as nn
import config, dataset, model, engine, utils
from transformers import AdamW, get_linear_schedule_with_warmup

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    
    set_seed(7) 

    df_train, df_test = utils.split_data()
    
    print(f'Experiment Name: {config.EXP_NAME}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device name:', torch.cuda.get_device_name(0))

    train_dataset = dataset.FakeNewsDataset(df_train)
    
#     train_data_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
#                             shuffle=True)
    ## using padding
    train_data_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                            shuffle=True, collate_fn=dataset.MyCollate())

    

    val_dataset = dataset.FakeNewsDataset(df_test)
    
#     val_data_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE,
#                             shuffle=True)
    ## using padding
    val_data_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE,
                            shuffle=True, collate_fn=dataset.MyCollate())
    
    print("Dataset Loaded")
    
    model = model.Multiple_Images_Model()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    
    print("Model Loaded")
    
    optimizer = AdamW(model.parameters(), lr=config.LR)

    # Total number of training steps
    num_train_steps = len(train_data_loader) *config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_func_epoch(epoch, model, train_data_loader, device, optimizer, scheduler)
        val_loss, acc, prec, rec, f1_score = engine.eval_func(model, val_data_loader, device, epoch)

        utils.log_model(epoch, train_loss, val_loss, acc, prec, rec, f1_score)

        if (val_loss < best_loss) and (config.SAVE_MODEL == True):
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss
