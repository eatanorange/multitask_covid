import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.trainer import train_covid_model,train_rsna_model
from utils.logger import logger
writer = SummaryWriter(log_dir='runs/multitask')

########################################################################################################
logger.info('Start training...')
batchsizes=16
epochs=10
from model.multitask_resunet import model
from dataset_covid import covid_train_dataset,covid_val_dataset
from dataset_rsna import rsna_train_dataset,rsna_val_dataset
covid_train_dataloader = DataLoader(covid_train_dataset, batch_size=batchsizes, shuffle=True)
covid_val_dataloader = DataLoader(covid_val_dataset, batch_size=batchsizes, shuffle=False)
rsna_train_dataloader = DataLoader(rsna_train_dataset, batch_size=batchsizes, shuffle=True)
rsna_val_dataloader = DataLoader(rsna_val_dataset, batch_size=batchsizes, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)
########################################################################################################


criterion.cuda()
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
    train_rsna_model(model,rsna_train_dataloader,rsna_val_dataloader,criterion,criterion,0.1,optimizer,num_epoch=epoch,writer=writer,logger=logger)
    
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
    train_covid_model(model,covid_train_dataloader,covid_val_dataloader,criterion,optimizer,num_epoch=epoch,writer=writer,logger=logger)

writer.close()
