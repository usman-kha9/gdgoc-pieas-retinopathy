import os
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter

from dataset import RetinopathyDataset, get_transforms
from model import CustomCNN
from utils import compute_metrics, save_checkpoint, get_class_weights_from_labels, FocalLoss, plot_confusion_matrix

def make_sampler(df, label_col, num_samples):
    counts = Counter(df[label_col].tolist())
    num_classes = max(counts.keys()) + 1
    class_weights = get_class_weights_from_labels(df[label_col].tolist(), num_classes=num_classes)
    # For sampler we need sample weight per example: weight[class]
    weights = [class_weights[int(l)] for l in df[label_col].tolist()]
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return sampler, class_weights

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device).float()
        labels = labels.to(device).long()
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics, all_labels, all_preds

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device).float()
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics, all_labels, all_preds

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    df = pd.read_csv(args.csv_path)
    # initial split train / val / test (if not provided)
    train_df, rest_df = train_test_split(df, test_size=args.test_val_split, stratify=df[args.label_col], random_state=42)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, stratify=rest_df[args.label_col], random_state=42)

    os.makedirs(args.output_dir, exist_ok=True)
    train_csv = os.path.join(args.output_dir, 'train_split.csv')
    val_csv = os.path.join(args.output_dir, 'val_split.csv')
    test_csv = os.path.join(args.output_dir, 'test_split.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_transform = get_transforms(args.image_size, True)
    val_transform = get_transforms(args.image_size, False)

    train_ds = RetinopathyDataset(train_csv, args.img_dir, transform=train_transform, img_col=args.img_col, label_col=args.label_col)
    val_ds = RetinopathyDataset(val_csv, args.img_dir, transform=val_transform, img_col=args.img_col, label_col=args.label_col)
    test_ds = RetinopathyDataset(test_csv, args.img_dir, transform=val_transform, img_col=args.img_col, label_col=args.label_col)

    sampler = None
    class_weights = None
    if args.use_weighted_sampler:
        sampler, class_weights = make_sampler(train_df, args.label_col, num_samples=len(train_ds))
        print("Using WeightedRandomSampler with computed class weights:", class_weights)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = CustomCNN(num_classes=args.num_classes, in_channels=3, dropout=args.dropout).to(device)

    # choose loss
    weight_tensor = torch.tensor(class_weights).float().to(device) if class_weights is not None else None
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=weight_tensor)
        print("Using FocalLoss (gamma=", args.focal_gamma, ")")
    else:
        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    best_metric = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir else None
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    for epoch in range(1, args.epochs + 1):
        since = time.time()
        train_loss, train_metrics, train_labels, train_preds = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_metrics, val_labels, val_preds = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        is_best = val_metrics['f1'] > best_metric
        if is_best:
            best_metric = val_metrics['f1']
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_metric,
        }, is_best, checkpoint_dir=args.output_dir, filename=checkpoint_name)

        time_elapsed = time.time() - since
        print(f"Epoch {epoch}/{args.epochs}  time: {time_elapsed:.0f}s")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}")
        print(f"  Train F1: {train_metrics['f1']:.4f}  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}  Val Recall: {val_metrics['recall']:.4f}")

        # TensorBoard logs
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/train_f1', train_metrics['f1'], epoch)
            writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch)
            # confusion matrix on val
            fig = plot_confusion_matrix(val_labels, val_preds, figsize=(6,6))
            writer.add_figure('ConfusionMatrix/val', fig, epoch)
            plt_close = getattr(fig, 'clear', None)
            try:
                plt.close(fig)
            except Exception:
                pass

        if epochs_no_improve >= args.early_stopping:
            print("Early stopping triggered.")
            break

    # Save final best weights
    final_path = os.path.join(args.output_dir, 'final_best.pth')
    torch.save({'state_dict': best_model_wts, 'best_metric': best_metric}, final_path)
    print("Training complete. Best F1:", best_metric)
    print("Saved best model to:", final_path)

    # Evaluate on test set with best weights
    model.load_state_dict(best_model_wts)
    test_loss, test_metrics, test_labels, test_preds = validate(model, test_loader, criterion, device)
    print("Test Loss:", test_loss)
    print("Test Metrics:", test_metrics)
    if writer:
        writer.add_scalar('Metrics/test_f1', test_metrics['f1'], 0)
        fig = plot_confusion_matrix(test_labels, test_preds, figsize=(6,6))
        writer.add_figure('ConfusionMatrix/test', fig, 0)
        try:
            plt.close(fig)
        except Exception:
            pass
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='CSV with image paths and labels')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Where to save models and logs')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_val_split', type=float, default=0.2, help='portion to hold for val+test (split in half)')
    parser.add_argument('--early_stopping', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--img_col', type=str, default='image_path')
    parser.add_argument('--label_col', type=str, default='label')
    parser.add_argument('--use_weighted_sampler', type=bool, default=False)
    parser.add_argument('--use_focal', type=bool, default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--tensorboard_dir', type=str, default='runs')
    args = parser.parse_args()
    main(args)