import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#模型和数据集
from model.multitask_resunet import model
from dataset.dataset_covid import covid_train_dataset,covid_val_dataset
from dataset.dataset_rsna import rsna_train_dataset,rsna_val_dataset

# 数据加载器
batchsizes=16
epochs=4

covid_train_dataloader = DataLoader(covid_train_dataset, batch_size=batchsizes, shuffle=True)
covid_val_dataloader = DataLoader(covid_val_dataset, batch_size=batchsizes, shuffle=False)
rsna_train_dataloader = DataLoader(rsna_train_dataset, batch_size=batchsizes, shuffle=True)
rsna_val_dataloader = DataLoader(rsna_val_dataset, batch_size=batchsizes, shuffle=False)


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)

writer = SummaryWriter(log_dir='runs/multitask')

def train_covid_model(model, train_loader, val_loader,criterion, optimizer, num_epoch=1,writer=writer):
    model.train()
    best_accuracy=0
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    print('*****train covid*****')
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

    print(f'train Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')
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
    print(f'val Loss: {epoch_loss:.4f} - val Accuracy: {accuracy:.4f} - val Precision: {precision:.4f} - val Recall: {recall:.4f} - val F1 Score: {f1:.4f}')
    #保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        print(f'Best model saved with accuracy: {best_accuracy:.4f}')
    torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{num_epoch}.pth')
    print('-' * 50)


def train_rsna_model(model,train_loader, val_loader,criterion, optimizer, num_epoch=1,rate=0.1,writer=writer):
    model.train()
    best_accuracy=0
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    print('*****train rsna*****')
    for inputs, labels,masks in tqdm(train_loader, desc=f'Epoch {num_epoch+1}'):
        inputs=inputs.cuda()
        labels=labels.cuda()
        masks=masks.cuda()
        optimizer.zero_grad()
        outputs,outputs_seg,_ = model(inputs)#######################改这里
        loss1 = criterion(outputs, labels)
        loss2 = criterion(outputs_seg, masks)
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

    print(f'train Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')
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
            loss1 = criterion(outputs, labels)
            loss2 = criterion(outputs_seg, masks)
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
    print(f'val Loss: {epoch_loss:.4f} - val Accuracy: {accuracy:.4f} - val Precision: {precision:.4f} - val Recall: {recall:.4f} - val F1 Score: {f1:.4f}')
    #保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        print(f'Best model saved with accuracy: {best_accuracy:.4f}')
    torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{num_epoch}.pth')
    print('-' * 50)

for epoch in range(epochs):
    for param in model.classifier_covid.parameters():
            param.requires_grad = False
    for param in model.classifier_rsna.parameters():
            param.requires_grad = True
    for param in model.up1.parameters():
            param.requires_grad = True
    for param in model.up2.parameters():
            param.requires_grad = True
    for param in model.up3.parameters():
            param.requires_grad = True
    for param in model.up4.parameters():
            param.requires_grad = True
    for param in model.outc.parameters():
            param.requires_grad = True
    train_rsna_model(model,rsna_train_dataloader,rsna_val_dataloader,criterion,optimizer,num_epoch=epoch)
    
    for param in model.classifier_covid.parameters():
            param.requires_grad = True
    for param in model.classifier_rsna.parameters():
        param.requires_grad = False
    for param in model.up1.parameters():
        param.requires_grad = False
    for param in model.up2.parameters():
        param.requires_grad = False
    for param in model.up3.parameters():
        param.requires_grad = False
    for param in model.up4.parameters():
        param.requires_grad = False
    for param in model.outc.parameters():
        param.requires_grad = False
    train_covid_model(model,covid_train_dataloader,covid_val_dataloader,criterion,optimizer,num_epoch=epoch)

writer.close()