# sanity_grl.py 

# This script will take a toy example and apply gradient reversal 
'''
1. Define the source and target domain
    Source ~ N(-mu/2, 0.5)
    Target ~ N(mu/2, 0.5)
2. Have a domain classifier where I try to distinguish between source and target 
3. Add a contrastive term so that source and target are represented dissimilarly


The parameters the model will have are as follows: 
    - Input X: (1,1)
    - Encoder Z: Z = x - sign(x)*w, where w.shape = (1,1)
    - Domain classifier: z-> d*z -> sigmoid -> probability
'''

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from scipy.stats import ks_2samp

class Model(nn.Module):
    def __init__(self, w_init=0.5, d_init=1.0): 
        super().__init__()
        self.w = nn.Parameter(torch.tensor([[w_init]]))
        self.disc = nn.Linear(1,1, bias=False)
        self.disc.weight.data.fill_(d_init)

    def forward(self, x):
        z = x-torch.sign(x)*self.w
        domain_logit = self.disc(z)
        return domain_logit, z

def main():

    # Generate the data 
    mu = 2
    std = 0.5
    num_samples = 100

    source = np.random.normal(-mu/2, std, (num_samples, 1))
    target = np.random.normal( mu/2, std, (num_samples, 1))

    source_labels = np.zeros((num_samples,1))
    target_labels = np.ones((num_samples, 1))

    x      = torch.tensor(np.concatenate([source, target]), dtype=torch.float32)
    labels = torch.tensor(np.concatenate([source_labels, target_labels]), dtype=torch.float32)

    model     = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = 1000
    for epoch in range(n_epochs):
        domain_logit, z = model(x)
        loss = criterion(domain_logit, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            z_np  = z.detach().numpy().flatten()
            ks, _ = ks_2samp(z_np[:num_samples], z_np[num_samples:])
            print(f"Epoch {epoch+1:3d} | loss={loss.item():.4f} | w={model.w.item():.4f} | d={model.disc.weight.item():.4f} | KS={ks:.4f}")

if __name__ == "__main__":
    main()
