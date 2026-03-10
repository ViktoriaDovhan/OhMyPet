from csv import DictReader
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cv.model import ShelterPetResNet18, ANIMAL_TYPE_CLASSES, SIZE_CLASSES, COLOR_CLASSES

DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_PATH = DATASET_DIR / "labels.csv"
MODEL_OUTPUT = Path("cv/models/shelter_pet_resnet18_multitask.pth")

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001

animal_type_to_idx = {label: index for index, label in enumerate(ANIMAL_TYPE_CLASSES)}
size_to_idx = {label: index for index, label in enumerate(SIZE_CLASSES)}
color_to_idx = {label: index for index, label in enumerate(COLOR_CLASSES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class PetDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(record["image_path"]).convert("RGB")
        image = self.transform(image)

        animal_type = torch.tensor(record["animal_type"], dtype=torch.long)
        size = torch.tensor(record["size"], dtype=torch.long)
        color = torch.tensor(record["color"], dtype=torch.long)

        return image, animal_type, size, color

def load_records():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл: {LABELS_PATH}")

    records = []

    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = DictReader(file)

        for row in reader:
            filename = row["filename"].strip()
            animal_type = row["animal_type"].strip()
            size = row["size"].strip()
            color = row["color"].strip()

            if animal_type not in animal_type_to_idx:
                raise ValueError(f"Невідомий тип тварини: {animal_type}")

            if size not in size_to_idx:
                raise ValueError(f"Невідомий розмір: {size}")

            if color not in color_to_idx:
                raise ValueError(f"Невідоме забарвлення: {color}")

            image_path = IMAGES_DIR / filename

            if not image_path.exists():
                raise FileNotFoundError(f"Не знайдено зображення: {image_path}")

            records.append({
                "image_path": image_path,
                "animal_type": animal_type_to_idx[animal_type],
                "size": size_to_idx[size],
                "color": color_to_idx[color]
            })

    if len(records) < 2:
        raise ValueError("Для навчання потрібно щонайменше 2 зображення")

    return records

def split_records(records):
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(records), generator=generator).tolist()

    split_index = int(len(indices) * 0.8)

    if split_index == 0:
        split_index = 1

    if split_index >= len(indices):
        split_index = len(indices) - 1

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    return train_records, val_records

def evaluate(model, loader, criterion_type, criterion_size, criterion_color):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    correct_type = 0
    correct_size = 0
    correct_color = 0

    with torch.no_grad():
        for images, animal_type, size, color in loader:
            images = images.to(device)
            animal_type = animal_type.to(device)
            size = size.to(device)
            color = color.to(device)

            outputs = model(images)

            loss_type = criterion_type(outputs["animal_type"], animal_type)
            loss_size = criterion_size(outputs["size"], size)
            loss_color = criterion_color(outputs["color"], color)

            loss = loss_type + loss_size + loss_color

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred_type = outputs["animal_type"].argmax(dim=1)
            pred_size = outputs["size"].argmax(dim=1)
            pred_color = outputs["color"].argmax(dim=1)

            correct_type += (pred_type == animal_type).sum().item()
            correct_size += (pred_size == size).sum().item()
            correct_color += (pred_color == color).sum().item()

    avg_loss = total_loss / total_samples
    type_acc = correct_type / total_samples
    size_acc = correct_size / total_samples
    color_acc = correct_color / total_samples
    score = (type_acc + size_acc + color_acc) / 3

    return avg_loss, type_acc, size_acc, color_acc, score

def main():
    records = load_records()
    train_records, val_records = split_records(records)

    train_dataset = PetDataset(train_records, train_transform)
    val_dataset = PetDataset(val_records, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ShelterPetResNet18(pretrained=True).to(device)

    criterion_type = nn.CrossEntropyLoss()
    criterion_size = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = -1.0
    saved_any = False

    for epoch in range(EPOCHS):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for images, animal_type, size, color in train_loader:
            images = images.to(device)
            animal_type = animal_type.to(device)
            size = size.to(device)
            color = color.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss_type = criterion_type(outputs["animal_type"], animal_type)
            loss_size = criterion_size(outputs["size"], size)
            loss_color = criterion_color(outputs["color"], color)

            loss = loss_type + loss_size + loss_color
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples

        val_loss, type_acc, size_acc, color_acc, score = evaluate(
            model,
            val_loader,
            criterion_type,
            criterion_size,
            criterion_color
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"type_acc={type_acc:.4f} | "
            f"size_acc={size_acc:.4f} | "
            f"color_acc={color_acc:.4f}"
        )

        if score > best_score:
            best_score = score
            saved_any = True
            MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT)

    if not saved_any:
        MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUTPUT)

    print(f"Best score: {best_score:.4f}")
    print(f"Saved to: {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()