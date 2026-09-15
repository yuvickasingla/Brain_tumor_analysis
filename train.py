import argparse, os, csv, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
import utils
from data_loader import create_dataloaders
from model import get_model


class EarlyStopping:
    def __init__(self, patience: int = config.PATIENCE):
        self.patience  = patience
        self.counter   = 0
        self.best_acc  = 0.0
        self.triggered = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered



def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total



def _build_optimizer(model, model_name: str, phase: int):
  
    lr_head = config.LR.get(model_name, 1e-4)
    lr_bb   = config.LR_BACKBONE

    if model_name == 'custom_cnn' or phase == 1:
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head, weight_decay=1e-4
        )

    # Phase 2 — parameter groups
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fc' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.Adam([
        {'params': backbone_params, 'lr': lr_bb},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=1e-4)



def train(args):
    utils.seed_everything()
    device = utils.get_device()
    print(f"\n{'='*55}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    train_loader, val_loader, _, class_names = create_dataloaders(
        batch_size=args.batch_size
    )

    model     = get_model(args.model, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    save_path = config.model_save_path(args.model)

    start_epoch = 0
    if args.resume and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch} (val_acc={ckpt['val_acc']:.2f}%)")

    phase     = 1
    optimizer = _build_optimizer(model, args.model, phase)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=3)
    stopper   = EarlyStopping(patience=config.PATIENCE)

    log_path = os.path.join(config.LOGS_DIR, f'{args.model}_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'phase',
                                 'train_loss', 'train_acc',
                                 'val_loss',   'val_acc'])

    best_val_acc = 0.0
    phase2_started = False

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if (args.model != 'custom_cnn'
                and not phase2_started
                and epoch >= args.epochs // 2):
            print(f"\n[Epoch {epoch+1}] Switching to Phase 2 — unfreezing backbone...\n")
            model.unfreeze()
            phase     = 2
            optimizer = _build_optimizer(model, args.model, phase)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                          patience=3)
            stopper   = EarlyStopping(patience=config.PATIENCE)  # reset stopper
            phase2_started = True

        train_loss, train_acc = _run_epoch(model, train_loader, criterion,
                                           optimizer, device, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion,
                                           optimizer, device, train=False)
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:>3}/{args.epochs}  "
              f"Phase {phase}  "
              f"Train {train_acc:.2f}%  "
              f"Val {val_acc:.2f}%  "
              f"({elapsed:.0f}s)")

        # Log
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, phase,
                                    f'{train_loss:.4f}', f'{train_acc:.2f}',
                                    f'{val_loss:.4f}',   f'{val_acc:.2f}'])

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':              epoch,
                'model_name':         args.model,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':            val_acc,
                'class_names':        class_names,
            }, save_path)
            print(f"Saved best model  val_acc={val_acc:.2f}%")

        if stopper.step(val_acc):
            print(f"\nEarly stopping triggered after {config.PATIENCE} epochs without improvement.")
            break

    print(f"\nTraining complete.  Best val acc: {best_val_acc:.2f}%")
    print(f"Checkpoint: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain tumor model on BRISC2025')
    parser.add_argument('--model',       type=str, default='custom_cnn',
                        choices=['custom_cnn', 'resnet50', 'efficientnet','mobilenetv3'],
                        help='Model architecture to train')
    parser.add_argument('--epochs',      type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch-size',  type=int, default=config.BATCH_SIZE)
    parser.add_argument('--resume',      action='store_true',
                        help='Resume from existing checkpoint')
    args = parser.parse_args()
    train(args)
