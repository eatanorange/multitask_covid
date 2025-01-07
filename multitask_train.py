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
#test_loader = DataLoader(dataset, batch_size=batchsizes,shuffle=False)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)

def compute_iou(pred, target, num_classes):
    """
    计算IoU

    参数:
    pred: Tensor[N, H, W] 预测分割结果
    target: Tensor[N, H, W] 真实标签
    num_classes: int 类别数量

    返回:
    iou: Tensor[num_classes] 每个类别的IoU值
    """
    ious = []
    pred = pred.argmax(dim=1)
    pred = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    for c in range(num_classes):
        # 计算交集
        intersection = torch.sum(pred[:, c, :, :] * target[:, c, :, :])
        # 计算并集
        union = torch.sum(pred[:, c, :, :]) + torch.sum(target[:, c, :, :]) - intersection
        # 计算IoU
        iou = intersection / (union + 1e-6)  # 防止除以0
        ious.append(iou)

    return torch.tensor(ious)



# 训练函数
writer = SummaryWriter(log_dir='runs/multitask')
total_train_step=0
total_val_step=0
rate=0.1
for epoch in range(epochs):
     if epoch%2==0:
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
        
        model.train()
        train_bar=tqdm(rsna_train_dataloader)
        for data in train_bar:
            imgs,labels,masks=data
            imgs=imgs.cuda()
            labels=labels.cuda()
            masks=masks.cuda()
            output_classify,output_segment,output_classify2=model(imgs)
            loss1=criterion(output_classify,labels)
            loss2=criterion(output_segment,masks)
            loss=(1-rate)*loss1+rate*loss2

            optimizer.zero_grad()#清零梯度
            loss.backward()#反向传播,给一个梯度
            optimizer.step()#优化器更新,就是根据梯度进行梯度下降

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                writer.add_scalar("train_loss", loss.item(), total_train_step)
            train_bar.set_description("train_epoch:{}".format(epoch))
        
        model.eval()
        total_miou = 0
        test_bar=tqdm(rsna_val_dataloader)
        total_loss=0
        total_accuracy=0
        for data in test_bar:
            imgs,labels,masks=data
            imgs=imgs.cuda()
            labels=labels.cuda()
            masks=masks.cuda()
            output_classify,output_segment,output_classify2=model(imgs)
            #计算准确率
            accuracy = (output_classify.argmax(1) == labels).sum()
            total_accuracy=total_accuracy+accuracy

            loss1=criterion(output_classify,labels)
            loss2=criterion(output_segment,masks)
            loss=(1-rate)*loss1+rate*loss2
            total_loss=total_loss+loss.item()
            miou=compute_iou(output_segment, masks, 2)
            test_bar.set_description("val_epoch:{}".format(epoch))
            total_val_step = total_val_step + 1
        total_loss=total_loss/len(rsna_val_dataset)
        total_accuracy=total_accuracy/len(rsna_val_dataset)
        total_miou += miou
        total_miou /= len(rsna_val_dataloader)

        writer.add_scalar("val_loss", total_loss,epoch)
        print("第{}轮训练结束，验证集loss为{},正确率为{}，miou：{}".format(epoch,total_loss,total_accuracy,total_miou))
        torch.save(model.state_dict, "weight/model_{}.pth".format(epoch))
        print("模型已保存")
        epoch = epoch + 1

     else:
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
        model.train()
        train_bar=tqdm(covid_train_dataloader)
        for data in train_bar:
            imgs,labels=data
            imgs=imgs.cuda()
            labels=labels.cuda()
            output_classify2,output_segment,output_classify=model(imgs)
            loss=criterion(output_classify,labels)
            optimizer.zero_grad()#清零梯度
            loss.backward()#反向传播,给一个梯度
            optimizer.step()#优化器更新,就是根据梯度进行梯度下降

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                writer.add_scalar("train_loss", loss.item(), total_train_step)
            train_bar.set_description("train_epoch:{}".format(epoch))

        model.eval()
        test_bar=tqdm(covid_val_dataloader)
        total_loss=0
        total_accuracy=0
        for data in test_bar:
            imgs,labels=data
            imgs=imgs.cuda()
            labels=labels.cuda()
            
            output_classify2,output_segment,output_classify=model(imgs)
            #计算准确率
            accuracy = (output_classify.argmax(1) == labels).sum()
            total_accuracy=total_accuracy+accuracy

            loss=criterion(output_classify,labels)
            total_loss=total_loss+loss.item()
            test_bar.set_description("val_epoch:{}".format(epoch))
            total_val_step = total_val_step + 1
        total_loss=total_loss/len(covid_val_dataset)
        total_accuracy=total_accuracy/len(covid_val_dataset)

        writer.add_scalar("val_loss", total_loss,epoch)
        print("第{}轮训练结束，验证集loss为{},正确率为{}".format(epoch,total_loss,total_accuracy))
        torch.save(model.state_dict, "weight/model_{}.pth".format(epoch))
        print("模型已保存")
        epoch = epoch + 1

