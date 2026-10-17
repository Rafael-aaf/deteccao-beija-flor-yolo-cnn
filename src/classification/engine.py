import torch
import torch.nn as nn


def train_epoch(model, loader, criterion, optimizer, device, freeze_bn):
    model.train()
    if freeze_bn:
        # o BatchNorm do backbone congelado mantém as estatísticas do ImageNet
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += len(targets)
    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    probs, all_targets = [], []
    total_loss = 0.0
    for images, targets in loader:
        logits = model(images.to(device))
        if criterion is not None:
            total_loss += criterion(logits, targets.to(device)).item() * len(targets)
        probs.append(logits.softmax(dim=1).cpu())
        all_targets.append(targets)
    targets_np = torch.cat(all_targets).numpy()
    loss = total_loss / len(targets_np) if criterion is not None else None
    return loss, torch.cat(probs).numpy(), targets_np