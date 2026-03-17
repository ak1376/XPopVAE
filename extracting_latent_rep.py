# extracting_latent_rep.py

'''
This script will pass in genomic windows into the model and extract the latent representations. I then will compare the raw similarity of the windows
(through edit distance or whatever) to the similarity of the latent representations (through cosine similarity or whatever). This will give us a sense
of how well the model is learning to capture the structure of the data in its latent space.
'''

from pathlib import Path
import torch
import numpy as np 
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path("/sietch_colab/akapoor/XPopVAE")))
from src.model import ConvVAE

def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vae_config = checkpoint["vae_config"]
    input_length = int(checkpoint["input_length"])

    model = ConvVAE(
        input_length=input_length,
        in_channels=1,
        hidden_channels=vae_config["model"]["hidden_channels"],
        kernel_size=int(vae_config["model"]["kernel_size"]),
        stride=int(vae_config["model"]["stride"]),
        padding=int(vae_config["model"]["padding"]),
        latent_dim=int(vae_config["model"]["latent_dim"]),
        use_batchnorm=False,
        activation="elu",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def compute_manhattan_similarity_matrix(G: np.ndarray) -> np.ndarray:
    n, m = G.shape
    sim = np.empty((n, n), dtype=np.float32)

    for i in range(n):
        dists = np.abs(G[i] - G).sum(axis=1)
        sim[i, :] = 1.0 - dists / (2.0 * m)

    return sim

def extract_latent_representations(checkpoint_path: Path, genotype_npy: Path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(checkpoint_path, device)

    G_truth = np.load(genotype_npy)
    if G_truth.ndim != 2:
        raise ValueError(f"Expected genotype matrix of shape (n_individuals, n_snps), got {G_truth.shape}")

    # convert to tensor: (batch, channels, length)
    x = torch.tensor(G_truth, dtype=torch.float32).unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        h = model.encoder(x)                  # (N, C, L_enc)
        h_flat = h.view(h.size(0), -1)        # (N, C * L_enc)
        mu = model.fc_mu(h_flat)              # (N, latent_dim)
        logvar = model.fc_logvar(h_flat)      # (N, latent_dim)

    latent_mu = mu.cpu().numpy()
    latent_logvar = logvar.cpu().numpy()

    print("latent_mu shape:", latent_mu.shape)
    print("latent_logvar shape:", latent_logvar.shape)

    return latent_mu, latent_logvar

def get_upper_triangle(M):
    i, j = np.triu_indices_from(M, k=1)
    return M[i, j]

def bin_similarity(raw_sim, latent_sim, n_bins=20):
    raw_vals = get_upper_triangle(raw_sim)
    latent_vals = get_upper_triangle(latent_sim)

    # Define bins
    bins = np.linspace(raw_vals.min(), raw_vals.max(), n_bins + 1)

    bin_indices = np.digitize(raw_vals, bins) - 1  # bin ids

    bin_means = []
    bin_stds = []
    bin_centers = []

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) == 0:
            continue

        bin_means.append(latent_vals[mask].mean())
        bin_stds.append(latent_vals[mask].std())
        bin_centers.append((bins[b] + bins[b+1]) / 2)

    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)

def plot_binned_similarity_line(bin_centers, bin_means, bin_stds, outdir):
    plt.figure(figsize=(8, 5))

    plt.plot(bin_centers, bin_means, marker='o')
    plt.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        alpha=0.2
    )

    plt.xlabel("Raw similarity")
    plt.ylabel("Mean latent cosine similarity")
    plt.title("Structure preservation: raw vs latent")
    plt.grid(alpha=0.3)

    plt.savefig(outdir / "binned_similarity.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_path = '/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/vae/vae_blocklen__blocklen2/vae_outputs/checkpoints/final_model.pt'
    genotype_npy = '/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy'
    output_dir = Path('/sietch_colab/akapoor/XPopVAE/latent_sim_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device)

    print(f"Loading genotype matrix: {genotype_npy}")
    G_truth = np.load(genotype_npy)
    if G_truth.ndim != 2:
        raise ValueError(
            f"Expected genotype matrix of shape (n_individuals, n_snps), got {G_truth.shape}"
        )

    print(f"Genotype matrix shape: {G_truth.shape}")

    # Compute a similarity matrix for the raw genotype windows (e.g., using edit distance or Hamming distance)
    print("Computing raw similarity matrix...")
    raw_similarity = compute_manhattan_similarity_matrix(G_truth)
    np.save(Path(output_dir) / "raw_similarity.npy", raw_similarity)

    print("Extracting latent representations...")
    latent_rep_mu, latent_rep_logvar = extract_latent_representations(checkpoint_path, genotype_npy)

    print(f'Raw Similarity matrix shape: {raw_similarity.shape}')
    print(f'Latent representations shape: {latent_rep_mu.shape}')

    print("Computing latent similarity matrix...")
    from sklearn.metrics.pairwise import cosine_similarity
    latent_similarity = cosine_similarity(latent_rep_mu)
    np.save(Path(output_dir) / "latent_similarity.npy", latent_similarity)

    print(f'Latent Similarity matrix shape: {latent_similarity.shape}')

    # Now let's plot the raw similarltiy as a heatmap 
    plt.figure(figsize=(10, 8))
    plt.imshow(raw_similarity, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Raw Similarity Matrix (1 - Manhattan Distance)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.savefig(Path(output_dir) / "raw_similarity_heatmap.png")
    plt.show()

    # Now let's plot the latent similarity as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(latent_similarity, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Latent Similarity Matrix (Cosine Similarity)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.savefig(Path(output_dir) / "latent_similarity_heatmap.png")
    plt.show()

    # Now plot the correspondence between them
    bin_centers, bin_means, bin_stds = bin_similarity(raw_similarity, latent_similarity)
    plot_binned_similarity_line(bin_centers, bin_means, bin_stds, output_dir)


if __name__ == "__main__":
    main()


