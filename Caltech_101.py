import os
import copy
import shutil
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 数据集划分
def split_dataset(data_dir, train_dir, val_dir, test_dir, val_size=0.1, test_size=0.2):
    """
    将数据集划分为训练集、验证集和测试集。
    默认划分比例为：训练集70%，验证集10%，测试集20%。
    """
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 遍历每个类别
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path) and category != 'BACKGROUND_Google':
            category_train_dir = os.path.join(train_dir, category)
            category_val_dir = os.path.join(val_dir, category)
            category_test_dir = os.path.join(test_dir, category)
            os.makedirs(category_train_dir, exist_ok=True)
            os.makedirs(category_val_dir, exist_ok=True)
            os.makedirs(category_test_dir, exist_ok=True)

            # 随机划分训练集、验证集和测试集,
            images = os.listdir(category_path)
            random.shuffle(images)
            val_split = int(len(images) * val_size)
            test_split = int(len(images) * test_size)
            val_images = images[:val_split]
            test_images = images[val_split:val_split+test_split]
            train_images = images[val_split+test_split:]

            # 复制图片到新文件夹
            for img in train_images:
                shutil.copy(os.path.join(category_path, img), category_train_dir)
            for img in val_images:
                shutil.copy(os.path.join(category_path, img), category_val_dir)
            for img in test_images:
                shutil.copy(os.path.join(category_path, img), category_test_dir)


# 数据集预处理与加载
def load_dataset(data_dir, train_dir, val_dir, test_dir, batch_size=32):
    """
    加载数据集并进行预处理。
    预处理包括：数据增强、图像大小调整、张量转换、标准化。
    分批次加载数据集，默认batch_size=32。
    """

    if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
        # 数据集划分
        split_dataset(data_dir, train_dir, val_dir, test_dir)

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化，使用ImageNet数据集的均值和标准差
    ])
    
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变换
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #  获取类别权重
    # class_counts = torch.zeros(len(train_dataset.classes))
    # for _, labels in train_loader:
    #     class_counts += torch.bincount(labels, minlength=len(class_counts))
    # class_weights = 1. / class_counts
    # class_weights = class_weights / class_weights.sum()

    return train_loader, val_loader, test_loader


# 模型构建
def build_model(num_classes, pretrained=True):
    """
    构建ResNet-18模型。
    如果pretrained=True，则加载预训练的模型。
    如果pretrained=False，则加载不使用预训练权重的模型。
    """

    model_path = '替换为工作目录\\pretrained_resnet18.pth' # 为方便加载，可将预训练模型保存到本地

    if pretrained and os.path.exists(model_path):
        # 加载预训练的ResNet-18模型
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(model_path))
    elif pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), model_path)
    else:
        # 加载不使用预训练权重的ResNet-18模型
        model = models.resnet18(weights=None)

    # 修改输出层
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc = nn.Sequential(
       nn.Dropout(0.5),  # 添加Dropout层，防止过拟合
       nn.Linear(model.fc.in_features, num_classes) 
    )
    return model


# 模型训练
def train_model(model, train_loader, val_loader, device, 
                optimizer, criterion, scheduler=None, num_epochs=10,
                  writer=None, start_epoch=0, save_dir=None):
    """
    训练模型。
    训练包括：训练阶段、验证阶段、学习率调整。
    训练记录包括：训练损失、验证损失、验证准确率。
    可定义参数包括：优化器、损失函数、学习率调整策略、训练轮数。
    """
    
    model.to(device)
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 迭代训练
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # 乘以batch_size
        
        train_loss = running_loss / len(train_loader.dataset)

        # 学习率调整
        if scheduler:
            scheduler.step()

        # 记录训练损失到TensorBoard
        if writer:
            writer.add_scalar('Training Loss', train_loss, epoch + start_epoch)

        # 验证阶段
        val_acc, val_loss = validate_model(model, val_loader, device, criterion)

        # 记录验证损失和准确率到TensorBoard
        if writer:
            writer.add_scalar('Validation Loss', val_loss, epoch + start_epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch + start_epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 加载最佳模型并保存
    model.load_state_dict(best_model)
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    return model


# 模型验证
def validate_model(model, val_loader, device, criterion):
    """
    验证模型。
    验证包括：计算验证损失和准确率。
    """

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播，计算预测值和损失
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    accuracy = correct / total
    val_loss = running_loss / len(val_loader.dataset)

    return accuracy, val_loss


# 模型测试
def test_model(model, test_loader, device):
    """
    测试模型，计算测试准确率。
    被注释掉的代码可用于保存每个类别的测试结果到CSV文件。
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    # class_correct = defaultdict(int)
    # class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #         # 统计每个类别的测试结果
    #         for label, pred in zip(labels, predicted):
    #             class_total[label.item()] += 1
    #             if pred == label:
    #                 class_correct[label.item()] += 1

    # class_names = test_loader.dataset.classes
    # results = []
    # for i in range(101):
    #     accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
    #     results.append([
    #         class_names[i], 
    #         class_total[i], 
    #         class_correct[i], 
    #         f"{accuracy:.2f}%"
    #     ])
    
    # # 保存为CSV文件
    # with open('替换为工作目录\\class_results.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['类别', '样本数', '正确数', '准确率'])
    #     writer.writerows(results)

    accuracy = correct / total

    return accuracy


# 预训练模型的训练与测试
def train_test_pretrained_model(model, train_loader, val_loader, test_loader, device, 
                                lr=1e-3, weight_decay=1e-3, num_epochs_1=10, num_epochs_2=5, num_epochs_3=5,
                                  writer=None, save_dir=None):
    """
    分阶段训练预训练模型并测试。
    训练包括：冻结部分层、解冻部分层、解冻所有层。
    运行记录包括：训练损失、验证损失、验证准确率、测试准确率。
    可定义参数包括：学习率、权重衰减、各阶段的训练轮数。
    """
    
    criterion = nn.CrossEntropyLoss()
    print("Training with pre-trained model.")

    # 第一阶段，冻结所有预训练层，只训练输出层
    print("Stage 1: Freeze all layers except the output layer.")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # 采用AdamW优化器，学习率衰减采用CosineAnnealingLR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model = train_model(model, train_loader, val_loader, device, 
                        optimizer, criterion, scheduler, 
                        num_epochs=num_epochs_1, writer=writer, save_dir=save_dir)


    # 第二阶段，解冻部分层进行微调
    print("Stage 2: Unfreeze some layers and fine-tune the model.")

    for param in model.layer4.parameters():
        param.requires_grad = True   # 解冻layer4
    for param in model.layer3.parameters():
        param.requires_grad = True   # 解冻layer3

    # 采用AdamW优化器，学习率衰减采用StepLR
    optimizer = torch.optim.AdamW([
        {'params': model.fc.parameters()},
        {'params': model.layer3.parameters(), 'lr': lr*0.1},
        {'params': model.layer4.parameters(), 'lr': lr*0.1}
    ], lr=lr/2, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    model = train_model(model, train_loader, val_loader, device, 
                        optimizer, criterion, scheduler, 
                        num_epochs=num_epochs_2, writer=writer, start_epoch=num_epochs_1, save_dir=save_dir)


    # 第三阶段，解冻所有层进行微调
    print("Stage 3: Unfreeze all layers and fine-tune the model.")

    for param in model.parameters():
        param.requires_grad = True  # 解冻所有层

    # 采用SGD优化器，学习率衰减采用StepLR
    optimizer = torch.optim.SGD(model.parameters(), lr=lr*0.1, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    model = train_model(model, train_loader, val_loader, device, 
                        optimizer, criterion, scheduler,
                        num_epochs=num_epochs_3, writer=writer, start_epoch=num_epochs_1+num_epochs_2, save_dir=save_dir)

    print("Training finished.")

    # 测试模型
    accuracy = test_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


# 非预训练模型的训练与测试
def train_test_no_pretrained_model(model, train_loader, val_loader, test_loader, device, 
                                   lr=1e-2, weight_decay=1e-3, num_epochs=20,
                                     writer=None, save_dir=None):
    """
    训练非预训练模型并测试。
    """

    criterion = nn.CrossEntropyLoss()

    # 采用SGD优化器，学习率衰减采用StepLR
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    print("Training without pre-trained model.")
    model = train_model(model, train_loader, val_loader, device, 
                        optimizer, criterion, scheduler, 
                        num_epochs=num_epochs, writer=writer, save_dir=save_dir)

    print("Training finished.")

    accuracy = test_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':

    work_dir = '替换为工作目录' # 工作目录
    
    # 加载数据集
    data_dir = '替换为数据集所在目录\\101_ObjectCategories' # 数据集路径
    train_dir = os.path.join(work_dir, 'train') # 训练集路径
    val_dir = os.path.join(work_dir, 'val') # 验证集路径
    test_dir = os.path.join(work_dir, 'test') # 测试集路径
    save_dir = work_dir # 模型保存路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数调整
    model_0 = build_model(101, pretrained=True)

    batch_sizes = [16, 32, 64] # 批次大小
    learning_rates = [1e-2, 1e-3, 1e-4] # 学习率
    weight_decays = [1e-3, 1e-4, 1e-5] # 正则化系数

    # 网格搜索
    for batch_size in batch_sizes:
        # 加载数据集
        train_loader, val_loader, test_loader, class_weights = load_dataset(data_dir, train_dir, val_dir, test_dir, batch_size=batch_size)
        print("Dataset loaded.")

        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}")

                # 创建TensorBoard SummaryWriter
                log_dir = os.path.join(work_dir, f"batch_size_{batch_size}_lr_{learning_rate}_weight_decay_{weight_decay}")
                writer = SummaryWriter(log_dir)

                model = copy.deepcopy(model_0) # 复制预训练模型

                # 训练和测试模型
                train_test_pretrained_model(model, train_loader, val_loader, test_loader, device,
                                            lr=learning_rate, weight_decay=weight_decay, num_epochs_1=15, num_epochs_2=10, num_epochs_3=5,
                                            writer=writer, save_dir=save_dir)
                writer.close()


    # 对比实验
    # 加载数据集
    train_loader, val_loader, test_loader = load_dataset(data_dir, train_dir, val_dir, test_dir)
    print("Dataset loaded.")

    writer_pretrained = SummaryWriter(os.path.join(work_dir, "pretrained_resnet18"))
    writer_no_pretrained = SummaryWriter(os.path.join(work_dir, "no_pretrained_resnet18"))
    
    # 训练和测试预训练模型
    model = build_model(101, pretrained=True)
    train_test_pretrained_model(model, train_loader, val_loader, test_loader, device,
                                lr=1e-3, weight_decay=1e-3, num_epochs_1=10, num_epochs_2=5, num_epochs_3=5,
                                writer=writer_pretrained, save_dir=save_dir)

    # 训练和测试非预训练模型
    model = build_model(101, pretrained=False)
    train_test_no_pretrained_model(model, train_loader, val_loader, test_loader, device,
                                num_epochs=20, writer=writer_no_pretrained)

    # 关闭TensorBoard
    writer_pretrained.close()
    writer_no_pretrained.close()

