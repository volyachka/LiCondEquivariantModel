import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
from tqdm import tqdm
import wandb
import os

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import torch.nn.functional as F

def train_epoch(model, train_dataloader, optimizer, criterion, device):
    model.train()  

    total_train_loss = 0.0
    num_samples = 0

    y_true = []
    y_pred = []

    for data in tqdm(train_dataloader):
        predictions = []
        importances = []
        loss = 0.0
        num_samples += len(data)
        for structures_batch in data:
            structures_batch = structures_batch.to(device)
            outputs = model(structures_batch)
            predictions, importances = outputs[:, 0], outputs[:, 1]
            importances = F.softmax(importances, dim = 0)
            final_prediction = (importances * predictions).sum()
            loss += criterion(final_prediction, structures_batch['target'][0])

            y_true.append(structures_batch['target'][0].unsqueeze(0))
            y_pred.append(final_prediction.unsqueeze(0))

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()

    r2 = r2_score(y_true, y_pred)
    avg_val_loss = total_train_loss / num_samples

    return avg_val_loss, r2

def validate_epoch(model, val_dataloader, criterion, device):
    model.eval()  

    total_val_loss = 0.0
    num_samples = 0

    y_true = []
    y_pred = []

    for data in tqdm(val_dataloader):
        predictions = []
        importances = []
        loss = 0.0
        num_samples += len(data)
        for structures_batch in data:
            structures_batch = structures_batch.to(device)
            outputs = model(structures_batch)
            predictions, importances = outputs[:, 0], outputs[:, 1]
            importances = F.softmax(importances, dim = 0)
            final_prediction = (importances * predictions).sum()
            loss += criterion(final_prediction, structures_batch['target'][0])

            y_true.append(structures_batch['target'][0].unsqueeze(0))
            y_pred.append(final_prediction.unsqueeze(0))

        total_val_loss += loss.item()
        
    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()

    r2 = r2_score(y_true, y_pred)
    avg_val_loss = total_val_loss / num_samples

    return avg_val_loss, r2



def train(model, train_dataloader, val_dataloader, config):
    train_losses = []
    val_losses = []

    criterion_name = config['training']['criterion']
    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()

    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = config['training']['num_epochs']

    if config['training']['device'] == "" or config['training']['device'] is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['training']['device']

    model = model.to(device)

    project_name = config['wandb']['project_name']
    entity = config['wandb']['entity_name']
    run_name = config['experiment_name']
    if project_name is not None:

        wandb.init(entity = entity, project=project_name, name = run_name, config=config)

    for epoch in range(1, num_epochs + 1):
        avg_train_loss, r2_train = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(avg_train_loss)
        avg_val_loss, r2_val = validate_epoch(model, val_dataloader, criterion, device)
        val_losses.append(avg_val_loss)

        if project_name is not None:
            wandb.log({
                            "epoch": epoch,
                            "train_loss": avg_train_loss,
                            "r2_train": r2_train,
                            "r2_val": r2_val

                    })
            


        # if project_name is not None:
        #     val_thorough_loss, val_thorough_r2 = thorough_validation(model, val_dataloader, criterion, device)
    

            # if epoch % config['training']['save_model_every_n_epochs'] == 0:
            #     wandb.log({
            #             "val_thorough_loss": val_thorough_loss,
            #             "val_thorough_r2": val_thorough_r2

            #         })
            # else:
            #     wandb.log({
            #             "epoch": epoch,
            #             "train_loss": avg_train_loss,
            #             "val_loss": avg_val_loss,
            #             "train_r2": r2_train,
            #             "val_r2": r2_val
            #         })    
            
    #     if epoch % config['training']['save_model_every_n_epochs'] == 0:
    #         name = config['experiment_name']
    #         output_dir = config['output_dir']
    #         model_path = os.path.join(output_dir, f"{name}_model_epoch_{epoch}.pt")
    #         optimizer_path = os.path.join(output_dir, f"{name}_optimizer_epoch_{epoch}.pt")
    #         torch.save(model.state_dict(), model_path)
    #         torch.save(optimizer.state_dict(), optimizer_path)

    #     if config['training']['verbose'] == True:

    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    #         plt.figure(figsize=(15, 10))

    #         plt.subplot(1, 2, 1)
    #         plt.plot(range(1, epoch + 2), train_losses, label='Training Loss', color='blue')
    #         plt.title('Training Loss')
    #         plt.xlabel('Epoch')
    #         plt.ylabel('Loss')
    #         plt.legend()

    #         plt.subplot(1, 2, 2)
    #         plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='red')
    #         plt.title('Validation Loss')
    #         plt.xlabel('Epoch')
    #         plt.ylabel('Loss')
    #         plt.legend()
    #         plt.show()
        
    #     clear_output(True)
        
    # if project_name is not None:
    #     wandb.finish()

    # return model