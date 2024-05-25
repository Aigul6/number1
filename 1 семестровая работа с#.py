import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from tqdm import tqdm
import shutil
import zipfile

class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str, error_dir: str):
        super().__init__()
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2
        self.error_dir = error_dir

        with zipfile.ZipFile(train_cats_path, 'r') as zip_ref:
            self.dir1_list = sorted(zip_ref.namelist())[:1000]

        with zipfile.ZipFile(train_dogs_path, 'r') as zip_ref:
            self.dir2_list = sorted(zip_ref.namelist())[:1000]

    def getImageIndex(self, index):
        if index < len(self.dir1_list):
            img_bytes = self.readImageFromZip(self.path_dir1, self.dir1_list[index])
            id = 0
        else:
            img_bytes = self.readImageFromZip(self.path_dir2, self.dir2_list[index - len(self.dir1_list)])
            id = 1
        img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
        return img, id

    def readImageFromZip(self, zip_path, file_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(file_path) as file:
                return file.read()

    def moveImage(self, path):
        filename = os.path.basename(path)
        dst_path = os.path.join(self.error_dir, filename)
        if os.path.exists(dst_path):
            print(f"Файл уже существует в каталоге ошибок: {dst_path}")
        else:
            try:
                shutil.move(path, self.error_dir)
            except shutil.Error as e:
                print(f"Ошибка изображения: {path}")
                print(e)

    def __getitem__(self, index):
        img, id_class = self.getImageIndex(index)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)
        t_id = torch.tensor(id_class)

        return {'img': t_img, 'label': t_id}

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)


# Создание папки для ошибочных изображений
error_images_dir = "C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#\\1 семестровая работа с#\\error"
if not os.path.exists(error_images_dir):
    os.makedirs(error_images_dir)

# Пути к данным
train_cats_path = "C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#\\1 семестровая работа с#\\Cat.zip"
train_dogs_path = "C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#\\1 семестровая работа с#\\Dog.zip"
test_cats_path = "C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#\\1 семестровая работа с#\\Cat.zip"
test_dogs_path = "C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#\\1 семестровая работа с#\\Dog.zip"

# Создание наборов данных
train_ds_catsdogs = Dataset2class(train_cats_path, train_dogs_path, error_images_dir)
test_ds_catsdogs = Dataset2class(test_cats_path, test_dogs_path, error_images_dir)

# Параметры обучения
batch_size = 16
epochs = 10

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=True
)

# Модель
model = tv.models.resnet34(num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
loss_fn = loss_fn.to(device)

# Обучение модели
for epoch in range(epochs):
    model.train()
    for sample in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        img, label = sample['img'].to(device), sample['label'].to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

# Тестирование модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sample in tqdm(test_loader, desc='Testing'):
        img, label = sample['img'].to(device), sample['label'].to(device)
        pred = model(img)
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
