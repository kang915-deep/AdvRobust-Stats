import torchattacks

def get_attack(model, attack_type, eps, alpha=None):
    if attack_type.lower() == 'fgsm':
        return torchattacks.FGSM(model, eps=eps)
    elif attack_type.lower() == 'pgd':
        alpha = alpha if alpha is not None else eps / 4
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=10, random_start=True)
    else:
        raise ValueError(f"Unsupported attack: {attack_type}")
