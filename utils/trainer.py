import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_classifer_model(model, train_loader, val_loader,criterion, optimizer, num_epoch,writer,logger):
    model.train()
    best_accuracy=0
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    logger.info(f'*****Epoch {num_epoch+1} train*****')
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {num_epoch+1}'):
        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)#######################改这里
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _,preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f'train Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')
    writer.add_scalar('Train/Loss', epoch_loss, num_epoch)
    writer.add_scalar('Train/Accuracy', accuracy, num_epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], num_epoch)
    #val
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'val'):
            inputs=inputs.cuda()
            labels=labels.cuda()
            outputs = model(inputs)###################################################改这里
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f'val Loss: {epoch_loss:.4f} - val Accuracy: {accuracy:.4f} - val Precision: {precision:.4f} - val Recall: {recall:.4f} - val F1 Score: {f1:.4f}')
    #保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        logger.info(f'Best model saved with accuracy: {best_accuracy:.4f}')
    torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{num_epoch}.pth')
    logger.info('-' * 50)

def train_covid_model(model, train_loader, val_loader,criterion, optimizer, num_epoch,writer,logger):
    model.train()
    best_accuracy=0
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    logger.info(f'Epoch {num_epoch+1} *****train covid*****')
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {num_epoch+1}'):
        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        _,_,outputs = model(inputs)#######################改这里
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f'train Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')
    writer.add_scalar('Train/Loss', epoch_loss, num_epoch)
    writer.add_scalar('Train/Accuracy', accuracy, num_epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], num_epoch)
    #val
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'val'):
            inputs=inputs.cuda()
            labels=labels.cuda()
            _,_,outputs = model(inputs)###################################################改这里
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f'val Loss: {epoch_loss:.4f} - val Accuracy: {accuracy:.4f} - val Precision: {precision:.4f} - val Recall: {recall:.4f} - val F1 Score: {f1:.4f}')
    #保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        logger.info(f'Best model saved with accuracy: {best_accuracy:.4f}')
    torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{num_epoch}.pth')
    logger.info('-' * 50)


def train_rsna_model(model,train_loader, val_loader,criterion1,criterion2,rate, optimizer, num_epoch,writer,logger):
    model.train()
    best_accuracy=0
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    logger.info(f'*****Epoch {num_epoch+1} train rsna*****')
    for inputs, labels,masks in tqdm(train_loader, desc=f'Epoch {num_epoch+1}'):
        inputs=inputs.cuda()
        labels=labels.cuda()
        masks=masks.cuda()
        optimizer.zero_grad()
        outputs,outputs_seg,_ = model(inputs)#######################改这里
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs_seg, masks)
        loss=(1-rate)*loss1+rate*loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f'train Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')
    writer.add_scalar('Train/Loss', epoch_loss, num_epoch)
    writer.add_scalar('Train/Accuracy', accuracy, num_epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], num_epoch)
    #val
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels,masks in tqdm(val_loader, desc=f'val'):
            inputs=inputs.cuda()
            labels=labels.cuda()
            masks=masks.cuda()
            outputs,outputs_seg,_ = model(inputs)###################################改这里
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(outputs_seg, masks)
            loss=(1-rate)*loss1+rate*loss2

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f'val Loss: {epoch_loss:.4f} - val Accuracy: {accuracy:.4f} - val Precision: {precision:.4f} - val Recall: {recall:.4f} - val F1 Score: {f1:.4f}')
    #保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        logger.info(f'Best model saved with accuracy: {best_accuracy:.4f}')
    torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{num_epoch}.pth')
    logger.info('-' * 50)