import torch

def loss_sensivity(loader, model, loss_fn, device):

    LS = 0
    model.eval()

    for indx, (images, labels) in enumerate(loader):
        
        images, labels = images.to(device), labels.to(device)  
        images.requires_grad = True
        output = model(images)
        loss = loss_fn(output, labels)
        model.zero_grad()
        loss.backward() 
        data_grad = images.grad.data
        LS += data_grad.norm()  

    return LS
    