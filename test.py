import disent.dataset.sampling
import torch
from torch.utils.data import DataLoader

from ws_crl.dataset._groundtruth__sprites import SpritesData
from disent.dataset._base import DisentDataset
from disent.dataset.transform._transforms import ToImgTensorF32
sprites=SpritesData("../workspace/data",prepare=True)
dataset=DisentDataset(sprites,
                   sampler=disent.dataset.sampling.GroundTruthPairSampler(),
                   transform=ToImgTensorF32(),
                   return_indices=False,
                   return_factors=True
                   )
dataloader=torch.utils.data.DataLoader(dataset=dataset,shuffle=True,batch_size=32,num_workers=4)
for i,_ in enumerate(dataloader):
    print(_)
