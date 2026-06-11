import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.classifiers import RobustClassifier
from defense.statistical_pca import PCADefense
from tqdm import tqdm
import pickle
import os
import numpy as np

def get_dataloaders(dataset_name, batch_size=128):
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, num_classes

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, num_classes = get_dataloaders(args.dataset)
    
    model = RobustClassifier(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"Training {args.model} on {args.dataset} for {args.epochs} epochs.")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        
        model.eval()
        test_loss, t_correct, t_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                t_total += targets.size(0)
                t_correct += predicted.eq(targets).sum().item()
                
        print(f"Epoch: {epoch+1} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {100.*t_correct/t_total:.2f}%")
        
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{args.model}_{args.dataset}.pth")
    print("Model saved.")

def fit_pca(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, _, num_classes = get_dataloaders(args.dataset)
    
    model = RobustClassifier(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f"saved_models/{args.model}_{args.dataset}.pth", map_location=device))
    model.eval()
    
    print("Extracting features for PCA fitting...")
    features_list = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(trainloader, desc="Extracting features"):
            inputs = inputs.to(device)
            features = model.get_latent_features(inputs)
            features_list.append(features.cpu().numpy())
            
    features_np = np.concatenate(features_list, axis=0)
    
    print(f"Fitting PCA with {args.n_components} components...")
    pca_defense = PCADefense(n_components=args.n_components)
    pca_defense.fit(features_np)
    
    os.makedirs('saved_models', exist_ok=True)
    with open(f"saved_models/pca_{args.model}_{args.dataset}_{args.n_components}.pkl", "wb") as f:
        pickle.dump(pca_defense, f)
    print("PCA model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'fit_pca'])
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--n_components', type=int, default=128)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'fit_pca':
        fit_pca(args)
