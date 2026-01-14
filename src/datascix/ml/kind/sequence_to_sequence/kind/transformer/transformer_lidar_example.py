import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utilix.data.storage.kind.file.pkl.pkl import Pkl
from utilix.os.file_system.file.file import File
from utilix.os.file_system.path.path import Path


# =====================================================================================
#                               TRANSFORMER MODEL
# =====================================================================================

class TimeSeriesSeq2SeqTransformer(nn.Module):
    """
    Light encoder–decoder Transformer for time series forecasting.
    """

    def __init__(
            self,
            input_dim: int,
            model_dim: int,
            num_heads: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            max_src_length: int,
            max_tgt_length: int,
            dropout: float
    ) -> None:
        super().__init__()

        self._src_proj = nn.Linear(input_dim, model_dim)
        self._tgt_proj = nn.Linear(input_dim, model_dim)

        self._src_pos = nn.Embedding(max_src_length, model_dim)
        self._tgt_pos = nn.Embedding(max_tgt_length, model_dim)

        self._transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self._out = nn.Linear(model_dim, input_dim)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        batch, src_len, _ = src.shape
        _, tgt_len, _ = tgt.shape

        src_emb = self._src_proj(src)
        tgt_emb = self._tgt_proj(tgt)

        pos_src = torch.arange(src_len, device=src.device).unsqueeze(0)
        pos_tgt = torch.arange(tgt_len, device=tgt.device).unsqueeze(0)

        src_emb = src_emb + self._src_pos(pos_src)
        tgt_emb = tgt_emb + self._tgt_pos(pos_tgt)

        out = self._transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
        return self._out(out)


# =====================================================================================
#                               DATASET
# =====================================================================================

class LidarSequenceDataset(Dataset):
    """
    Creates sliding (src, tgt) windows from LiDAR sequence.
    """

    def __init__(self, scans: np.ndarray, src_len: int, tgt_len: int) -> None:
        self._scans = scans
        self._src_len = src_len
        self._tgt_len = tgt_len
        self._N = scans.shape[0] - src_len - tgt_len

        if self._N <= 0:
            raise ValueError("Not enough time steps.")

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> dict:
        src = self._scans[idx:idx + self._src_len]
        tgt = self._scans[idx + self._src_len:idx + self._src_len + self._tgt_len]
        return {
            "src": torch.from_numpy(src).float(),
            "tgt": torch.from_numpy(tgt).float()
        }


# =====================================================================================
#                               MASK
# =====================================================================================

def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


# =====================================================================================
#                               LOAD + FIX LIDAR (NO NORMALIZATION)
# =====================================================================================

def load_lidar_scan_vectors() -> np.ndarray:
    path = Path(
        "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/"
        "mind/memory/long_term/explicit/episodic/normal/"
        "lidar_scan_ranges_sliced_from_1_to_300000/"
        "lidar_scan_ranges_sliced_from_1_to_300000.pkl"
    )

    os_file = File.init_from_path(path)
    pk = Pkl(os_file, False)
    sliced = pk.load()
    vals = sliced.get_values()

    scans = []
    for v in vals:
        c = (
            v.get_formatted_data()
            .get_vector_representation()
            .get_components()
        )
        scans.append(c)

    scans = np.array(scans, dtype=np.float32)

    max_range = 15.0

    # Replace NaN/Inf with max range
    non_finite = ~np.isfinite(scans)
    scans[non_finite] = max_range

    # Clip to sensor range (0–15 m)
    scans = np.clip(scans, 0.0, max_range)

    # Use first 50k scans for speed (if more exist)
    if scans.shape[0] > 50000:
        scans = scans[:50000]

    print("Loaded LiDAR scans:", scans.shape)
    return scans


# =====================================================================================
#                               MAIN
# =====================================================================================

def main() -> None:
    src_len = 16
    tgt_len = 16
    batch_size = 32
    epochs = 5
    model_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scans = load_lidar_scan_vectors()
    input_dim = scans.shape[1]

    dataset = LidarSequenceDataset(scans, src_len, tgt_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TimeSeriesSeq2SeqTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_src_length=32,
        max_tgt_length=32,
        dropout=0.1
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    tgt_mask = generate_square_subsequent_mask(tgt_len, device)

    # ---------- Training ----------
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0

        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            dec_in = torch.zeros_like(tgt)
            dec_in[:, 1:, :] = tgt[:, :-1, :]

            optim.zero_grad()
            out = model(src, dec_in, tgt_mask)
            loss = loss_fn(out, tgt)

            if not torch.isfinite(loss):
                print("Non-finite loss encountered.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total += loss.item()
            n += 1

        print(f"Epoch {ep + 1}/{epochs}, train_loss={total / n:.6f}")

    # ---------- Evaluation: average Euclidean distance per prediction step ----------
    model.eval()

    # We want, for each horizon step k in [0..tgt_len-1],
    # the average Euclidean distance ||pred_k - tgt_k||_2 over all windows.
    step_error_sum = np.zeros(tgt_len, dtype=np.float64)
    step_count = np.zeros(tgt_len, dtype=np.int64)

    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for batch in eval_loader:
            src = batch["src"].to(device)  # (B, src_len, D)
            tgt = batch["tgt"].to(device)  # (B, tgt_len, D)

            dec_in = torch.zeros_like(tgt)
            dec_in[:, 1:, :] = tgt[:, :-1, :]

            pred = model(src, dec_in, tgt_mask)  # (B, tgt_len, D)

            diff = pred - tgt  # (B, tgt_len, D)
            # Euclidean distance over D for each (B, step)
            # result: (B, tgt_len)
            per_step_dist = torch.sqrt(torch.sum(diff * diff, dim=2))

            # Accumulate over batch
            per_step_dist_np = per_step_dist.cpu().numpy()
            step_error_sum += per_step_dist_np.sum(axis=0)
            step_count += per_step_dist_np.shape[0]

    avg_step_error = step_error_sum / step_count

    # ---------- Plot single curve ----------
    steps = np.arange(1, tgt_len + 1)

    plt.figure()
    plt.plot(steps, avg_step_error, marker="o")
    plt.xlabel("Prediction step (1 = 1st future scan, 16 = 16th)")
    plt.ylabel("Average Euclidean distance over all beams")
    plt.title("Average prediction error vs horizon (over all windows)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
