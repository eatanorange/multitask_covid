import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#模型和数据集
from model.singletask_resnet50 import model
from dataset.dataset_covid import covid_train_dataset,covid_val_dataset


# 数据加载器
batchsizes=32
epochs=300

train_loader = DataLoader(covid_train_dataset, batch_size=batchsizes, shuffle=True)
val_loader = DataLoader(covid_val_dataset, batch_size=batchsizes, shuffle=False)
#test_loader = DataLoader(dataset, batch_size=batchsizes,shuffle=False)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)

# 训练函数
Writer = SummaryWriter(log_dir='runs/covid_experiment')
def train_model(model, train_loader, val_loader,criterion, optimizer, num_epochs=25,writer=Writer):
    best_accuracy=0
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs=inputs.cuda()
            labels=labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
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
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        # 你也可以在这里添加对测试集的评估
        evaluate_model(model, val_loader, criterion)
        #test_model(model, test_loader, criterion)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoint/best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')
        torch.save(model.state_dict(), 'checkpoint/'+f'model_epoch_{epoch}.pth')
        print('-' * 50)
    writer.close()

# 评估函数
def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'val'):
            inputs=inputs.cuda()
            labels=labels.cuda()
            outputs = model(inputs)
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



# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs,writer=Writer)