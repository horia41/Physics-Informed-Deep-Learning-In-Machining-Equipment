import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch

from config import LABELS_CSV, SETS_DIR, TRAIN_SETS, VAL_SETS, TEST_SETS

def load_labels():
    df = pd.read_csv(LABELS_CSV)
    
    # keep only rows that have both an image and a wear value
    df = df[df['ImageName'].notna() & df['wear'].notna()]
    
    # reset index so rows are numbered 0, 1, 2...
    df = df.reset_index(drop=True)
    
    print(f"Total usable rows: {len(df)}")
    print(f"Sets present: {sorted(df['Set'].unique().tolist())}")
    
    return df


class ToolWearDataset(Dataset):
    def __init__(self, df, sets, transform=None):
        # filter to only the sets we want
        self.df = df[df['Set'].isin(sets)].reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        # tells PyTorch how many samples are in this dataset
        return len(self.df)
    
    def __getitem__(self, idx):
        # PyTorch calls this to get one sample by index
        row = self.df.iloc[idx]
        
        # load the image
        img_path = SETS_DIR / f"Set{row['Set']}" / "images" / row['ImageName']
        image = Image.open(img_path).convert('RGB')
        
        # get the wear label
        wear = torch.tensor(row['wear'], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, wear  # ← always returned as a pair

if __name__ == "__main__":
    df = load_labels()
    print(df.head())
    
    image_transformation = transforms.Compose([
    transforms.Resize((224, 224)),        # simple resize
    transforms.ToTensor(),                # convert to tensor [0,1]
    ])  

    train_dataset = ToolWearDataset(df, sets=TRAIN_SETS, transform=image_transformation)
    val_dataset   = ToolWearDataset(df, sets=VAL_SETS, transform=image_transformation)
    test_dataset  = ToolWearDataset(df, sets=TEST_SETS, transform=image_transformation)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    # test that one sample loads correctly
    image, wear = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Wear value:  {wear}")

    