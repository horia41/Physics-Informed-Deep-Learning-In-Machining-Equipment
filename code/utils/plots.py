from pathlib import Path
import matplotlib.pyplot as plt

def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")