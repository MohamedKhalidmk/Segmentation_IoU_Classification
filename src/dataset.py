import os
from collections import Counter

def load_dataset_with_mask_offset(images_dir, masks_dir):
    dataset = []
    classes = sorted(os.listdir(images_dir))
    print("Detected classes:", classes)

    for cls in classes:
        img_folder = os.path.join(images_dir, cls)
        mask_folder = os.path.join(masks_dir, cls)

        img_files = sorted(os.listdir(img_folder))
        mask_files = sorted(os.listdir(mask_folder))

        if len(img_files) != len(mask_files):
            print(f"Warning: Image/mask count mismatch in class {cls}")

        for i in range(len(img_files)):
            dataset.append({
                'class': cls,
                'img_path': os.path.join(img_folder, img_files[i]),
                'mask_path': os.path.join(mask_folder, mask_files[i])
            })

    print(f"Total images loaded: {len(dataset)}")
    return dataset

def class_distribution(dataset):
    class_counts = Counter([sample['class'] for sample in dataset])
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
    return class_counts
