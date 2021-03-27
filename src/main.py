import torch
import pandas as pd
import numpy as np
import random 
from torch.utils.data import DataLoader
import config, dataset, model, engine, utils

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":

	set_seed(42) 

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_dataset = dataset.FakeNewsDataset(df_train)
	train_data_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
	                        shuffle=True)

	val_dataset = dataset.FakeNewsDataset(df_test)
	val_data_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_sIZE,
	                        shuffle=True)

	model = model.Multiple_Images_Model()
	model.to(device)

	best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_func_epoch(epoch, train_data_loader, model, device, optimizer, scheduler)
        val_loss, acc, prec, rec, f1_score = engine.eval_func(epoch, model, val_data_loader, device)

        utils.log_model(epoch, train_loss, val_loss, acc, prec, rec, f1_score)

        if (val_loss < best_loss) and (config.SAVE_MODEL == True):
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss
