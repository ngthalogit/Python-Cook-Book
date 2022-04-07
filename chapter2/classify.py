from model import Net
import torch 
import numpy as np 

def deploy(model, dataset, weights_path, device, num_cls=2, sanity_check=False):
    len_data = len(dataset)
    y_hat = torch.zeros(len_data, num_cls)
    targets = np.zeros((len_data),dtype='uint8')
    
    elapsed_times = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            targets[i] = y 
            start = time().time
            y_hat[i] = model(x.unsqueeze(0).to(device))
            elapsed = time.time() - start
            elapsed_times.append(elapsed)   
            

