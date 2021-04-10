import torch
import numpy as np
from tqdm import tqdm
import utils

from sklearn import metrics

def train_func_epoch(epoch, model, data_loader, device, optimizer, scheduler):

    # Put the model into the training mode
    model.train()

    total_loss = 0

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Training- Epoch {epoch}")

            img_ip , text_ip, label = batch["img_ip"], batch["text_ip"], batch['label']

            ## Load the inputs to the device
            input_ids, attn_mask = tuple(t.to(device) for t in text_ip)
            img_ip = img_ip.to(device)
            label = label.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return Multimodal vec and total loss.
            sim_vec, logits_l2, logits_l3, logits_l4 = model(text=[input_ids, attn_mask], image=img_ip, label=label)
            
            del input_ids
            del attn_mask
            del img_ip
            
            loss = utils.final_loss(sim_vec, logits_l2, logits_l3, logits_l4, label)
            
            total_loss += loss.item()
            
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Update parameters and the learning rate
            loss.backward()
            optimizer.step()
            scheduler.step()

            single_epoch.set_postfix(train_loss=total_loss/(step+1))

    return total_loss / len(data_loader)


def eval_func(model, data_loader, device, epoch=1):

    # Put the model into the training mode
    model.eval()

    total_loss = 0
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0

    with tqdm(data_loader, unit="batch", total=len(data_loader)+1) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            img_ip , text_ip, label = batch["img_ip"], batch["text_ip"], batch['label']

            ## Load the inputs to the device
            input_ids, attn_mask = tuple(t.to(device) for t in text_ip)
            img_ip = img_ip.to(device)
            label = label.to(device)

            with torch.no_grad():
                sim_vec, logits_l2, logits_l3, logits_l4 = model(text=[input_ids, attn_mask], image=img_ip, label=label)
            
            del input_ids
            del attn_mask
            del img_ip
            
            loss = utils.final_loss(sim_vec, logits_l2, logits_l3, logits_l4, label)

            total_loss += loss.item()

            single_epoch.set_postfix(loss=loss.item())

            # Finding predictions 
            pred_multimodal = torch.argmax(logits_l4, dim=1).flatten().cpu().numpy()
#             pred_text = torch.argmax(logits_l2, dim=1).flatten().cpu().numpy()
#             pred_vis = torch.argmax(logits_l3, dim=1).flatten().cpu().numpy()

            # Find performance 
            tn, fp, fn, tp = metrics.confusion_matrix(label.cpu().numpy(), pred_multimodal, labels=[0,1]).ravel()

            total_tn += tn
            total_fp += fp 
            total_fn += fn
            total_tp += tp 


    acc, prec, rec, f1_score = utils.model_metric(total_tn, total_fp, total_fn, total_tp)
        
    print(f'Epoch:{epoch}, val_loss={total_loss/len(data_loader)}, accuracy={acc}, precision={prec}, recall={rec}, f1_score={f1_score}')

    return total_loss/len(data_loader), acc, prec, rec, f1_score
