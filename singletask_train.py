import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.trainer import train_classifer_model
from utils.logger import logger
writer = SummaryWriter(log_dir='runs/covid_experiment')


########################################################################################################################
logger.info('数据增强，resnet最后的线性层改了一下，adam0.0001，做完这个就做adam调参')
batchsizes=32
epochs=300
from model.singletask_smlp import model
from dataset_covid import covid_train_dataset,covid_val_dataset
covid_train_dataloader = DataLoader(covid_train_dataset, batch_size=batchsizes, shuffle=True)
covid_val_dataloader = DataLoader(covid_val_dataset, batch_size=batchsizes, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
########################################################################################################################

criterion.cuda()
for epoch in range(epochs):
    train_classifer_model(model, covid_train_dataloader, covid_val_dataloader, criterion, optimizer, epoch, writer, logger)

writer.close()