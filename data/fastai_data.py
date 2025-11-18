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
    Prepare ChestX-ray14 dataframe using official train/test split

    Args:
        data_dir: Path to data directory
        seed: Random seed
        filter_normal: If True, filter out "No Finding" images

    Returns:
        train_val_df: Training + validation dataframe (from official train split)
        disease_labels: List of disease labels
        test_df: Test dataframe (from official test split)
    """
    disease_labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]

    # Load official train/test split files
    print("Loading official train/test split...")
    train_list = pd.read_csv(f'{data_dir}/train_val_list.txt', header=None)
    train_list.columns = ['Image_Index']
    test_list = pd.read_csv(f'{data_dir}/test_list.txt', header=None)
    test_list.columns = ['Image_Index']

    print(f"  Official train: {len(train_list)} images")
    print(f"  Official test: {len(test_list)} images")

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

    # Convert Finding_Labels to list
    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(
        lambda s: [l for l in str(s).split('|')]
    )

    # Map image paths
    num_glob = glob(f'{data_dir}/*/images/*.png')
    img_path = {os.path.basename(x): x for x in num_glob}
    labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)
    labels_df = labels_df.dropna(subset=['Paths'])

    # Use official split
    train_val_df = labels_df[labels_df['Image_Index'].isin(train_list['Image_Index'])]
    test_df = labels_df[labels_df['Image_Index'].isin(test_list['Image_Index'])]

    # Filter normal if requested (for Phase 1 training)
    # Apply AFTER split to maintain consistent test set
    if filter_normal:
        print("Filtering out 'No Finding' images...")
        train_val_df = train_val_df[train_val_df['Finding_Labels'].apply(lambda x: 'No Finding' not in x)]
        test_df = test_df[test_df['Finding_Labels'].apply(lambda x: 'No Finding' not in x)]

    print(f"\nDataset prepared:")
    print(f"  Train+Val: {len(train_val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Unique patients (train+val): {train_val_df['Patient_ID'].nunique()}")
    print(f"  Unique patients (test): {test_df['Patient_ID'].nunique()}")

    return train_val_df, disease_labels, test_df


def get_validation_split(train_val_df, valid_pct=0.1, seed=85):
    """
    Get validation image indices from full train_val dataframe
    Split by PATIENT ID to avoid data leakage
    This should be called ONCE with full (unfiltered) train data

    Args:
        train_val_df: Full training dataframe (before any filtering)
        valid_pct: Validation percentage
        seed: Random seed

    Returns:
        val_image_indices: Set of Image_Index values for validation
    """
    np.random.seed(seed)

    # Split by PATIENT ID to avoid leakage
    unique_patients = train_val_df['Patient_ID'].unique()
    n_val_patients = int(len(unique_patients) * valid_pct)

    # Shuffle patients
    patient_indices = np.arange(len(unique_patients))
    np.random.shuffle(patient_indices)

    # Select validation patients
    val_patient_indices = patient_indices[:n_val_patients]
    val_patients = unique_patients[val_patient_indices]

    # Get all images from validation patients
    val_mask = train_val_df['Patient_ID'].isin(val_patients)
    val_image_indices = set(train_val_df[val_mask]['Image_Index'].values)

    print(f"\nGlobal validation split created (patient-level):")
    print(f"  Total patients: {len(unique_patients)}")
    print(f"  Validation patients: {len(val_patients)} ({len(val_patients)/len(unique_patients)*100:.1f}%)")
    print(f"  Total train images: {len(train_val_df)}")
    print(f"  Validation images: {len(val_image_indices)} ({len(val_image_indices)/len(train_val_df)*100:.1f}%)")
    print(f"  → No patient appears in both train and validation!")

    return val_image_indices


def create_dataloaders(train_val_df, disease_labels, batch_size=64, valid_pct=0.1, seed=85,
                       val_image_indices=None):
    """
    Create FastAI DataLoaders with consistent validation set

    Args:
        train_val_df: Training + validation dataframe
        disease_labels: List of disease labels
        batch_size: Batch size
        valid_pct: Validation percentage (default 0.1 = 10%)
        seed: Random seed
        val_image_indices: Optional set of image indices to use as validation
                          If provided, uses explicit split instead of RandomSplitter
                          This ensures consistent validation set across phases

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

    # Create splitter based on whether val_image_indices is provided
    if val_image_indices is not None:
        # Use explicit split for consistent validation across phases
        val_mask = train_val_df['Image_Index'].isin(val_image_indices)
        train_mask = ~val_mask

        def consistent_splitter(items):
            # items is the dataframe passed to dataloaders
            train_idx = list(np.where(train_mask.values)[0])
            val_idx = list(np.where(val_mask.values)[0])
            return train_idx, val_idx

        splitter = consistent_splitter
        print(f"\nUsing consistent validation split:")
        print(f"  Validation images (from global): {val_mask.sum()}")
    else:
        # Use RandomSplitter (default behavior)
        splitter = RandomSplitter(valid_pct=valid_pct, seed=seed)
        print(f"\nUsing random validation split (valid_pct={valid_pct})")

    # Create DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
        splitter=splitter,
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
    print(f"  Training samples: {len(dls.train_ds)}")
    print(f"  Validation samples: {len(dls.valid_ds)}")
    print(f"  Number of classes: {len(disease_labels)}")

    return dls


if __name__ == '__main__':
    # Test data preparation
    data_dir = '/kaggle/input/data'

    # Phase 1: Abnormal images only
    print("="*60)
    print("PHASE 1: Preparing abnormal images only")
    print("="*60)
    train_val_df_phase1, disease_labels,_ = prepare_chestxray14_dataframe(
        data_dir, seed=85, filter_normal=True
    )
    dls_phase1 = create_dataloaders(train_val_df_phase1, disease_labels, batch_size=64)

    # Phase 2: All images
    print("\n" + "="*60)
    print("PHASE 2: Preparing all images")
    print("="*60)
    train_val_df_phase2, disease_labels, _ = prepare_chestxray14_dataframe(
        data_dir, seed=85, filter_normal=False
    )
    dls_phase2 = create_dataloaders(train_val_df_phase2, disease_labels, batch_size=128)
