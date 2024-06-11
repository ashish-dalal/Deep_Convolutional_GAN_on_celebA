import torch 
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## HYPERPARAMETERS
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 5
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    ## SETTING UP TRANSFORMS
    transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ])

    dataset = torchvision.datasets.ImageFolder(root='data/celeb_dataset', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    ## SETTING UP OPTIMIZER
    optimizer_gen = torch.optim.Adam(params=gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_disc = torch.optim.Adam(params=disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    ## SETTING UP FIXED NOISE
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    writer_real = SummaryWriter(f"runs/DCGAN_MNIST/real")
    writer_fake = SummaryWriter(f"runs/DCGAN_MNIST/fake")

    step=0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_example, _) in enumerate(dataloader):
            real_example = real_example.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            
            ## TRAIN DISCRIMINATOR: max log(D(real_example)) + log(1-D(G(z)))
            
            #### discriminator prediction on real examples
            disc_pred_on_real = disc(real_example).reshape(-1)
            loss_disc_real = criterion(disc_pred_on_real, torch.ones_like(disc_pred_on_real))
            
            #### discriminator prediction on generated (fake) examples
            fake_example = gen(noise)
            disc_pred_on_fake = disc(fake_example)
            loss_disc_fake = criterion(disc_pred_on_fake, torch.zeros_like(disc_pred_on_fake))
            
            #### total loss of discriminator
            loss_disc = (loss_disc_fake + loss_disc_real) / 2
            
            #### discriminator zero grad
            disc.zero_grad()
            
            #### discriminator backprop
            loss_disc.backward(retain_graph=True)
            
            #### discriminator optimizer step
            optimizer_disc.step()
            
            ## TRAIN GENERATOR: min log(1-D(G(z))) <--> max log(D(G(z)))
            
            #### discriminator 
            output = disc(fake_example).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            
            #### generator zero grad
            gen.zero_grad()
            
            #### generator backprop
            loss_gen.backward()
            
            #### generator optimizer step
            optimizer_gen.step()
            
            if batch_idx%250 == 0:
                print(
                    f"Epoch[{epoch} / {NUM_EPOCHS}] \ step[{step}] "
                    f"Loss Discriminator: {loss_disc:.4f}, Loss Generator: {loss_gen:.4f}"
                )
                
                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_fake  = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real  = torchvision.utils.make_grid(real_example[:32], normalize=True)
                    
                    writer_fake.add_image(
                        "MNIST Fake Images", img_grid_fake, global_step=step
                    )
                    
                    writer_real.add_image(
                        "MNIST Real Images", img_grid_real, global_step=step
                    )
                    
                step += 1

    model_path = 'model/generator_DCGAN_celebA.pth'
    torch.save(gen.state_dict(), model_path)

    model_path = 'model/discriminator_DCGAN_celebA.pth'
    torch.save(disc.state_dict(), model_path)

if __name__ == '__main__':
    main()