import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from models.classifiers import RobustClassifier
from utils.attack import get_attack
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        transform_test = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    model = RobustClassifier(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f"saved_models/{args.model}_{args.dataset}.pth", map_location=device))
    model.eval()
    
    with open(f"saved_models/pca_{args.model}_{args.dataset}_{args.n_components}.pkl", "rb") as f:
        pca_defense = pickle.load(f)
        
    epsilons = args.epsilons
    baseline_accs = []
    pca_accs = []
    
    def test(model_to_test, atk=None):
        correct = 0
        total = 0
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            if atk is not None:
                inputs = atk(inputs, targets)
            outputs = model_to_test(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total

    print("Evaluating...")
    for eps in epsilons:
        print(f"Epsilon: {eps}")
        if eps == 0:
            b_acc = test(model)
            model.set_defense(pca_defense)
            p_acc = test(model)
            model.set_defense(None)
        else:
            atk = get_attack(model, args.attack, eps=eps)
            
            # Generate adv examples and test on baseline
            correct_b = 0
            correct_p = 0
            total = 0
            
            for inputs, targets in tqdm(testloader, leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs_adv = atk(inputs, targets)
                
                # Test Baseline
                outputs_b = model(inputs_adv)
                _, pred_b = outputs_b.max(1)
                correct_b += pred_b.eq(targets).sum().item()
                
                # Test PCA Defense
                model.set_defense(pca_defense)
                outputs_p = model(inputs_adv)
                _, pred_p = outputs_p.max(1)
                correct_p += pred_p.eq(targets).sum().item()
                model.set_defense(None)
                
                total += targets.size(0)
                
            b_acc = correct_b / total
            p_acc = correct_p / total
            
        baseline_accs.append(b_acc)
        pca_accs.append(p_acc)
        print(f"Baseline: {b_acc:.4f}, PCA: {p_acc:.4f}")
        
    plt.figure()
    plt.plot(epsilons, baseline_accs, marker='o', label='Baseline')
    plt.plot(epsilons, pca_accs, marker='s', label='PCA Defense')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('Robustness Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_curve.png')
    
    print("Generating Variance Distribution...")
    model.set_defense(None)
    clean_features = []
    adv_features = []
    
    atk = get_attack(model, args.attack, eps=epsilons[-1] if epsilons[-1] > 0 else 0.05)
    
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        f_clean = model.get_latent_features(inputs)
        clean_features.append(f_clean.detach().cpu().numpy())
        
        inputs_adv = atk(inputs, targets)
        f_adv = model.get_latent_features(inputs_adv)
        adv_features.append(f_adv.detach().cpu().numpy())
        break 
        
    clean_features = np.concatenate(clean_features, axis=0)
    adv_features = np.concatenate(adv_features, axis=0)
    
    clean_var = pca_defense.get_explained_variance(clean_features)
    adv_var = pca_defense.get_explained_variance(adv_features)
    
    plt.figure()
    plt.plot(clean_var, label='Clean Features')
    plt.plot(adv_var, label='Adversarial Features')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance')
    plt.yscale('log')
    plt.legend()
    plt.title('Variance Distribution')
    plt.savefig('variance_distribution.png')
    print("Saved robustness_curve.png and variance_distribution.png.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--attack', type=str, default='pgd')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.0, 0.01, 0.02, 0.04, 0.08])
    parser.add_argument('--n_components', type=int, default=128)
    
    args = parser.parse_args()
    evaluate(args)
