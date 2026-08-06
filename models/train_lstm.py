"""
PreViral — Fast LSTM Training (Checkpoint-Resumable)
Uses synthetic YouTube-style trajectory data (no large CSV needed).
Trains in ~3-5 minutes, saves every epoch so restarts don't lose progress.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob, re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
CKPT_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_checkpoint.pt")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── Generate realistic trajectory data ───────────────────────────────────────
def make_dataset(n=30000):
    """
    Synthesize YouTube-style trajectory data with realistic distributions.
    Uses real YouTube CSVs if available, otherwise generates realistic synthetic data.
    """
    YOUTUBE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase2",
                               "data", "raw_datasets", "youtube_trending")
    csvs = glob.glob(os.path.join(YOUTUBE_DIR, "*.csv"))

    if csvs:
        import pandas as pd
        dfs = []
        for f in csvs[:3]:
            try:
                df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', nrows=20000)
                dfs.append(df)
            except Exception:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined.columns = [c.lower().strip() for c in combined.columns]
            view_col = next((c for c in combined.columns if 'view' in c), None)
            title_col = next((c for c in combined.columns if 'title' in c), None)
            id_col    = next((c for c in combined.columns if c.startswith('video') or c == 'id'), None)
            if view_col and id_col:
                combined[view_col] = pd.to_numeric(combined[view_col], errors='coerce').fillna(0)
                titles, trajectories = [], []
                for vid_id, grp in combined.groupby(id_col):
                    views = grp[view_col].values
                    if len(views) < 2: continue
                    v1  = float(views[0])
                    v3  = float(views[min(2, len(views)-1)])
                    v7  = float(views[min(5, len(views)-1)])
                    v10 = float(views[min(8, len(views)-1)])
                    title = str(grp[title_col].iloc[0]) if title_col else ""
                    titles.append(title)
                    trajectories.append([v1, v3, v7, v10])
                if len(trajectories) > 1000:
                    print(f"  Loaded {len(trajectories):,} real YouTube trajectories")
                    return titles, np.array(trajectories, dtype=np.float32)

    # Synthetic fallback
    print("  Generating synthetic trajectory data...")
    np.random.seed(42)
    n_viral  = int(n * 0.15)
    n_good   = int(n * 0.30)
    n_normal = n - n_viral - n_good

    def make_traj(base, shape):
        v1  = base * np.random.uniform(0.05, 0.20, len(base))
        v3  = base * np.random.uniform(0.25, 0.60, len(base))
        v7  = base * np.random.uniform(0.60, 0.90, len(base))
        v10 = base * np.random.uniform(0.80, 1.00, len(base))
        return np.stack([v1, v3, v7, v10], axis=1) * shape

    base_viral  = np.random.lognormal(13, 1.2, n_viral)
    base_good   = np.random.lognormal(10, 1.0, n_good)
    base_normal = np.random.lognormal(7,  1.0, n_normal)

    trajs = np.concatenate([
        make_traj(base_viral,  np.random.uniform(0.9, 1.1, (n_viral, 4))),
        make_traj(base_good,   np.random.uniform(0.8, 1.0, (n_good, 4))),
        make_traj(base_normal, np.random.uniform(0.7, 0.9, (n_normal, 4))),
    ], axis=0)

    title_templates = [
        "How to {} like a pro in 2026",
        "You won't believe {} — shocking results",
        "I tried {} for 30 days — here's what happened",
        "The best {} guide for beginners",
        "{} is changing everything — here's why",
    ]
    topics = ["AI", "Python", "cooking", "fitness", "travel", "investing",
              "coding", "design", "crypto", "productivity"]
    titles = [
        title_templates[i % 5].format(topics[i % 10])
        for i in range(n)
    ]
    print(f"  Generated {n:,} synthetic trajectories")
    return titles, trajs.astype(np.float32)


def extract_features(titles):
    X = []
    for title in titles:
        t = str(title)[:300]
        s = _vader.polarity_scores(t)
        X.append([
            s['compound'], max(0, s['compound']), abs(s['compound']),
            s['pos'], s['neg'],
            min(len(t), 300) / 300,
            float('?' in t), float('!' in t),
            min(t.count('!'), 5) / 5,
            float(bool(re.search(r'how to|tutorial', t, re.I))),
            float(bool(re.search(r'best|top|worst', t, re.I))),
            float(bool(re.search(r'\d+', t))),
            float(bool(re.search(r'you|your|we', t, re.I))),
            float(bool(re.search(r'secret|hack|never|always', t, re.I))),
            float(bool(re.search(r'shorts|short|quick', t, re.I))),
            float(bool(re.search(r'full|complete', t, re.I))),
            float(bool(re.search(r'vs|versus', t, re.I))),
            float(bool(re.search(r'new|first|exclusive', t, re.I))),
            min(len(re.findall(r'[A-Z]', t[:50])), 10) / 10,
            min(len(t.split()), 20) / 20,
        ])
    return np.array(X, dtype=np.float32)


# ── Model ────────────────────────────────────────────────────────────────────
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=20, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden, hidden, layers,
                            dropout=dropout if layers > 1 else 0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        p = self.proj(x).unsqueeze(1).repeat(1, 4, 1)
        out, _ = self.lstm(p)
        return self.head(out).squeeze(-1)  # (batch, 4)


class TrajDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Train ────────────────────────────────────────────────────────────────────
def train():
    print("="*55)
    print("PreViral — LSTM Trajectory Model")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    titles, trajs = make_dataset(n=30000)

    print("Extracting features...")
    X_raw = extract_features(titles)

    # Log-normalize targets
    log_trajs = np.log1p(trajs)
    tmax = log_trajs.max(axis=0) + 1e-8
    y_norm = log_trajs / tmax

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y_norm, test_size=0.15, random_state=42)

    tr_dl = DataLoader(TrajDataset(X_tr, y_tr), batch_size=512, shuffle=True)
    va_dl = DataLoader(TrajDataset(X_val, y_val), batch_size=1024)

    model = TrajectoryLSTM().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loss_fn = nn.HuberLoss(delta=0.1)

    # Resume from checkpoint if it exists
    start_epoch = 1
    best_val = float('inf')
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float('inf'))
        print(f"Resuming from epoch {start_epoch} (best_val={best_val:.4f})")

    patience, pat_count = 10, 0
    EPOCHS = 40

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        tr_loss = sum(
            loss_fn(model(Xb.to(device)), yb.to(device)).item()
            for Xb, yb in tr_dl
        ) / len(tr_dl)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for Xb, yb in va_dl:
                va_loss += loss_fn(model(Xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_dl)
        sched.step()

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            pat_count = 0
        else:
            pat_count += 1

        # Save checkpoint every epoch (resilient to restarts)
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'opt': opt.state_dict(), 'best_val': best_val
        }, CKPT_PATH)
        if is_best:
            torch.save(model.state_dict(),
                       os.path.join(MODEL_SAVE_DIR, "trajectory_lstm_best.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS}: train={tr_loss:.4f}  val={va_loss:.4f}  {'*' if is_best else ''}")

        if pat_count >= patience:
            print(f"  Early stopping at epoch {epoch}. Best val={best_val:.4f}")
            break

    # Save final artifacts
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "trajectory_lstm.pt"))
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "trajectory_scaler.joblib"))
    joblib.dump(tmax,   os.path.join(MODEL_SAVE_DIR, "trajectory_target_max.joblib"))

    # Quick test
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(X[:1], dtype=torch.float32).to(device)
        pred = model(sample).cpu().numpy()[0]
    views = np.expm1(pred * tmax).astype(int)
    print(f"\n  Sample: Day1={views[0]:,}  Day3={views[1]:,}  Day7={views[2]:,}  Day10={views[3]:,}")
    print(f"\n  LSTM complete. Best val loss: {best_val:.4f}")
    print(f"  Saved: {MODEL_SAVE_DIR}/trajectory_lstm_best.pt")


if __name__ == "__main__":
    train()
