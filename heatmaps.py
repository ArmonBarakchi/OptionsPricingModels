import numpy as np
import matplotlib.pyplot as plt

from option_pricing import BlackScholesModel


def _price_call_put_black_scholes(S, K, days_to_maturity, r, vol):
    """Return (call, put) using your existing BlackScholesModel class."""
    m = BlackScholesModel(S, K, days_to_maturity, r, vol)
    return (
        float(m.calculate_option_price("Call Option")),
        float(m.calculate_option_price("Put Option")),
    )


def make_option_heatmaps_fig(
    K: float,
    days_to_maturity: int,
    r: float,
    spot_min: float,
    spot_max: float,
    vol_min: float,
    vol_max: float,
    spot_points: int = 10,
    vol_points: int = 10,
):
    """
    Creates a 1x2 figure: Call heatmap and Put heatmap for a grid of (S, vol).
    Uses matplotlib only.
    """
    # Defensive guards
    spot_min, spot_max = float(min(spot_min, spot_max)), float(max(spot_min, spot_max))
    vol_min, vol_max = float(min(vol_min, vol_max)), float(max(vol_min, vol_max))
    spot_points = int(max(2, spot_points))
    vol_points = int(max(2, vol_points))

    S_grid = np.linspace(spot_min, spot_max, spot_points)
    vol_grid = np.linspace(vol_min, vol_max, vol_points)

    call = np.zeros((vol_points, spot_points))
    put = np.zeros((vol_points, spot_points))

    for i, v in enumerate(vol_grid):
        for j, S in enumerate(S_grid):
            c, p = _price_call_put_black_scholes(S, K, days_to_maturity, r, v)
            call[i, j] = c
            put[i, j] = p

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    extent = [S_grid[0], S_grid[-1], vol_grid[0], vol_grid[-1]]

    im0 = axes[0].imshow(call, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("CALL")
    axes[0].set_xlabel("Spot Price")
    axes[0].set_ylabel("Volatility")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(put, origin="lower", aspect="auto", extent=extent)
    axes[1].set_title("PUT")
    axes[1].set_xlabel("Spot Price")
    axes[1].set_ylabel("Volatility")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    xmin, xmax = S_grid[0], S_grid[-1]
    ymin, ymax = vol_grid[0], vol_grid[-1]
    dx = (xmax - xmin) / call.shape[1]
    dy = (ymax - ymin) / call.shape[0]

    for ax, Z in [(axes[0], call), (axes[1], put)]:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                x = xmin + (j + 0.5) * dx
                y = ymin + (i + 0.5) * dy
                ax.text(
                    x, y, f"{Z[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=9, color="white"
                )

    fig.tight_layout()
    return fig
