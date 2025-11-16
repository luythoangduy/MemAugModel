"""
FastAI DataBlock and DataLoaders for ChestX-ray14
Follows the exact structure from mem_model.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.vision.all import *
from tqdm import tqdm
from glob import glob  # Import after fastai to avoid conflict


def prepare_chestxray14_dataframe(data_dir, seed=85, filter_normal=False):
    """
    Prepare ChestX-ray14 dataframe following notebook structure

    Args:
        data_dir: Path to data directory
        seed: Random seed
        filter_normal: If True, filter out "No Finding" images

    Returns:
        train_val_df: Training + validation dataframe
        disease_labels: List of disease labels
    """
    disease_labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]

    # Load labels
    labels_df = pd.read_csv(f'{data_dir}/Data_Entry_2017.csv')
    labels_df.columns = [
        'Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
        'Patient_Age', 'Patient_Gender', 'View_Position',
        'Original_Image_Width', 'Original_Image_Height',
        'Original_Image_Pixel_Spacing_X',
        'Original_Image_Pixel_Spacing_Y', 'dfd'
    ]

    # One hot encoding
    print("One-hot encoding disease labels...")
    for disease in tqdm(disease_labels):
        labels_df[disease] = labels_df['Finding_Labels'].map(
            lambda result: 1 if disease in result else 0
        )

    # Filter normal if requested (for Phase 1 training)
    if filter_normal:
        labels_df = labels_df[labels_df.Finding_Labels != 'No Finding']

    # Convert Finding_Labels to list
    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(
        lambda s: [l for l in str(s).split('|')]
    )

    # Map image paths
    num_glob = glob(f'{data_dir}/*/images/*.png')
    img_path = {os.path.basename(x): x for x in num_glob}
    labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)
    labels_df = labels_df.dropna(subset=['Paths'])

    # Split by patient ID (80% train+val, 20% test)
    unique_patients = np.unique(labels_df['Patient_ID'])
    train_val_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    train_val_df = labels_df[labels_df['Patient_ID'].isin(train_val_patients)]
    test_df = labels_df[labels_df['Patient_ID'].isin(test_patients)]
    print(f"\nDataset prepared:")
    print(f"  Total images: {len(labels_df)}")
    print(f"  Train+Val: {len(train_val_df)}")
    print(f"  Test: {len(labels_df) - len(train_val_df)}")
    print(f"  Unique patients (train+val): {len(train_val_patients)}")

    return train_val_df, disease_labels, test_df


def create_dataloaders(train_val_df, disease_labels, batch_size=64, valid_pct=0.125, seed=85):
    """
    Create FastAI DataLoaders following notebook structure

    Args:
        train_val_df: Training + validation dataframe
        disease_labels: List of disease labels
        batch_size: Batch size
        valid_pct: Validation percentage (default 0.125 = 12.5%)
        seed: Random seed

    Returns:
        dls: FastAI DataLoaders
    """
    # Item transforms
    item_transforms = [
        Resize((224, 224)),
    ]

    # Batch transforms
    batch_transforms = [
        Flip(),
        Rotate(),
        Normalize.from_stats(*imagenet_stats),
    ]

    # Get functions
    def get_x(row):
        return row['Paths']

    def get_y(row):
        labels = row[disease_labels].tolist()
        return labels

    # Create DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        get_x=get_x,
        get_y=get_y,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms
    )

    # Create DataLoaders
    dls = dblock.dataloaders(train_val_df, bs=batch_size)

    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(dls.train)}")
    print(f"  Validation batches: {len(dls.valid)}")
    print(f"  Number of classes: {len(disease_labels)}")

    return dls


if __name__ == '__main__':
    # Test data preparation
    data_dir = '/kaggle/input/data'

    # Phase 1: Abnormal images only
    print("="*60)
    print("PHASE 1: Preparing abnormal images only")
    print("="*60)
    train_val_df_phase1, disease_labels = prepare_chestxray14_dataframe(
        data_dir, seed=85, filter_normal=True
    )
    dls_phase1 = create_dataloaders(train_val_df_phase1, disease_labels, batch_size=64)

    # Phase 2: All images
    print("\n" + "="*60)
    print("PHASE 2: Preparing all images")
    print("="*60)
    train_val_df_phase2, disease_labels = prepare_chestxray14_dataframe(
        data_dir, seed=85, filter_normal=False
    )
    dls_phase2 = create_dataloaders(train_val_df_phase2, disease_labels, batch_size=128)
