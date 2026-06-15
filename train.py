import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import SimpleCNN

#超参数
EPOCHS=20
BATCH_SIZE=64
LR=0.001

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    total_loss=0

    for data,label in loader:
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)

def evaluate(model,loader,device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data,label in loader:
            data,label=data.to(device),label.to(device)
            outputs=model(data)
            predicted=torch.argmax(outputs,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()

    return 100*correct/total

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    #数据
    train_loader,test_loader=get_dataloaders(batch_size=BATCH_SIZE)
    #模型
    model=SimpleCNN().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    #训练
    for epoch in range(EPOCHS):
        avg_loss=train_one_epoch(model,train_loader,criterion, optimizer, device)
        accuracy=evaluate(model,test_loader,device)
        print(f"Epoch {epoch+1:>2}/{EPOCHS}  loss: {avg_loss:.4f}  acc: {accuracy:.2f}%")
    #保存权重
    torch.save(model.state_dict(),'results/simplecnn_cifar10.pth')
    print("Model saved to results/simplecnn_cifar10.pth ")

if __name__=='__main__':
    main()