import torch
import os
from Train.checkpoint import save_checkpoint, load_checkpoint


def train_one_epoch(model, loader, optimizer, step_fn, device):

    totals = {}

    model.train()

    for batch in loader:

        optimizer.zero_grad()

        loss, metrics = step_fn(model, batch, device)

        loss.backward()

        optimizer.step()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)

    avg_loss = {k: v / len(loader) for k, v in totals.items()}

    return avg_loss


def evaluate(model, val_loader, optimizer, eval_fn, device):

    val_metric = {}

    model.eval()
    with torch.no_grad():

        for batch in val_loader:
            metrics = eval_fn(model, batch, device)

            for k, v in metrics.items():
                val_metric[k] = val_metric.get(k, 0.0) + float(v)

    avg_val_metric = {k: v / len(val_loader) for k, v in val_metric.items()}

    return avg_val_metric


def train_epochs(
    model,
    train_loader,
    optimizer,
    step_fn,
    epochs,
    ckpt_interval,
    out_dir,
    device,
    val_loader=None,
    eval_fn=None,
    best_metric="loss",
    best_mode="min",  # "min" or "max"
    is_ssl=True,
):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    encoder_path = os.path.join(ckpt_dir, "encoder_only.pt")

    # resume
    start_epoch = load_checkpoint(model, optimizer, last_path)

    best_score = float("inf") if best_mode == "min" else float("-inf")

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, step_fn, device)

        val_metrics = {}
        if (val_loader is not None) and (eval_fn is not None):
            val_metrics = evaluate(model, val_loader, eval_fn, device)

        # choose score: prefer val metric if available, else train
        score = val_metrics.get(best_metric, train_metrics.get(best_metric))
        if score is None:
            raise ValueError(f"best_metric='{best_metric}' not found in metrics.")

        improved = (score < best_score) if best_mode == "min" else (score > best_score)
        if improved:
            best_score = score
            if is_ssl:
                torch.save(model.Encoder.state_dict(), encoder_path)
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                best_path,
            )

        if (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                last_path,
            )

        print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}", flush=True)
