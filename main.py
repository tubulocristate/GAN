import torch
from networks import GENERATOR, DISCRIMINATOR, GENERATOR_CONV, DISCRIMINATOR_CONV
import figures_creator 
import matplotlib.pyplot as plt
import config
import sys

device = torch.device("cuda")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    torch.autograd.set_detect_anomaly(True)
    G = GENERATOR_CONV().to(device)
    D = DISCRIMINATOR_CONV().to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    optimizer_G = torch.optim.AdamW(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.AdamW(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    square = figures_creator.make_square().unsqueeze(0).to(device)
    circle = figures_creator.make_circle().unsqueeze(0).to(device)
    sin = figures_creator.make_sinusoidal_image().unsqueeze(0).to(device)

    
    # TRAINING
    min_grads, max_grads = [], []
    for i in range(1001):
        for k in range(2):
            optimizer_D.zero_grad()
            noise = torch.rand(size=(100, 1, 1), device=device)
            discriminator_loss_square = -torch.log(D(square)) - torch.log(1 - D(G(noise)))
            #discriminator_loss_circle = -torch.log(D(circle)) - torch.log(1 - D(G(noise)))
            #discriminator_loss_sin = -torch.log(D(sin)) - torch.log(1 - D(G(noise)))
            #discriminator_loss = (discriminator_loss_square + discriminator_loss_circle + discriminator_loss_sin)/3
            discriminator_loss = discriminator_loss_square
            #discriminator_loss = discriminator_loss_sin
            discriminator_loss.backward()
            optimizer_D.step()

        optimizer_G.zero_grad()
        noise = torch.rand(size=(100, 1, 1), device=device)
        generator_loss = -torch.log(D(G(noise)))
        generator_loss.backward()
        optimizer_G.step()
        
        #min_grads.append(torch.min(G.linear_relu_sequence[2].weight.grad))
        #max_grads.append(torch.max(G.linear_relu_sequence[2].weight.grad))
        if i % 100 == 0:
            image = G(noise)
            image = image.cpu().detach().numpy()
            figures_creator.display_image(image[0])

    #plt.plot(min_grads, label="min")
    #plt.plot(max_grads, label="max")
    #plt.legend()
    #plt.show()

    return 0

if __name__ == "__main__":
    main()
    print("Done!")
