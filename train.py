import torch 
from utils import save_checkpoint, save_some_examples 
import config
from achitecture import Generator
import torch.nn as nn
from dataset_generator import MapDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


dataset = MapDataset()
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = Generator().to(config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()
l_scaler = torch.cuda.amp.GradScaler()
save_some_examples(model, data_loader, 'evaluations')

for epochs in range(config.EPOCHS):
    for idx, (x, y) in tqdm(enumerate(data_loader)):
        x = x.permute((0, 3, 1, 2))
        y_fake = model(x)

        loss = criterion(y_fake, y.permute((0, 3, 1, 2)))

        l_scaler.scale(loss)
        loss.backward()
        l_scaler.step(optimizer)
        l_scaler.update()
        optimizer.zero_grad()
    
    if epochs % 10 == 0 :
        save_checkpoint(model, optimizer, 'generator.pth.tar')
        save_some_examples(model, data_loader, epochs, 'evaluations')
