from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Root folder containing "etfs" and "stocks"
ROOT = Path(r"C:\Users\wisem\OneDrive\Desktop\Ahmad\Python Projects\archive")


def load_ticker(symbol: str, category: str) -> pd.DataFrame:
    """
    category: 'stocks' or 'etfs'
    symbol: ticker without .csv, e.g. 'A' or 'AAAU'
    """
    file_path = ROOT / category / f"{symbol}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep only rows with valid adjusted close
    df = df[df["Adj Close"] > 0].copy()

    # Derived quantities
    df["Log Price"] = np.log(df["Adj Close"])
    df["Daily Return %"] = df["Adj Close"].pct_change() * 100
    df["Log Return"] = df["Log Price"].diff()

    return df


def plot_ticker(df: pd.DataFrame, symbol: str, category: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Adjusted price
    axes[0].plot(df["Date"], df["Adj Close"], label="Adj Close")
    axes[0].set_title(f"{symbol} ({category})")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True)

    # Log-price
    axes[1].plot(df["Date"], df["Log Price"], label="Log Price")
    axes[1].set_ylabel("Log-Price")
    axes[1].legend()
    axes[1].grid(True)

    # Volume
    axes[2].plot(df["Date"], df["Volume"], label="Volume")
    axes[2].set_ylabel("Volume")
    axes[2].legend()
    axes[2].grid(True)

    # Log return
    axes[3].plot(df["Date"], df["Log Return"], label="Log Return")
    axes[3].set_ylabel("Log Return")
    axes[3].set_xlabel("Date")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def print_summary(df: pd.DataFrame, symbol: str) -> None:
    print(f"\nSummary for {symbol}")
    print("-" * 40)
    print(f"Rows: {len(df)}")
    print(f"Start date: {df['Date'].min().date()}")
    print(f"End date:   {df['Date'].max().date()}")
    print(f"Min Adj Close: {df['Adj Close'].min():.2f}")
    print(f"Max Adj Close: {df['Adj Close'].max():.2f}")
    print(f"Average Volume: {df['Volume'].mean():.0f}")
    print("\nFirst 5 rows:")
    print(df.head())

def compute_windowed_msd(df, window_size=252, max_lag=30, demean=True):
    """
    Compute MSD of log-price in moving windows.

    window_size : number of trading days in each window
    max_lag     : maximum lag (in trading days) for MSD
    demean      : if True, remove mean log-return within each window first

    Returns
    -------
    window_dates : list of timestamps (window center dates)
    lags         : np.ndarray of lags
    msd_matrix   : np.ndarray with shape (n_windows, max_lag)
    """
    x_full = df["Log Price"].to_numpy()
    dates = df["Date"].to_numpy()

    lags = np.arange(1, max_lag + 1)
    window_dates = []
    msd_list = []

    for start in range(0, len(df) - window_size + 1):
        end = start + window_size
        x = x_full[start:end].copy()

        if demean:
            r = np.diff(x)
            r = r - np.nanmean(r)
            x = np.concatenate([[0.0], np.cumsum(r)])

        msd = np.empty(max_lag, dtype=float)

        for i, lag in enumerate(lags):
            dx = x[lag:] - x[:-lag]
            msd[i] = np.mean(dx**2)

        center_idx = start + window_size // 2
        window_dates.append(dates[center_idx])
        msd_list.append(msd)

    msd_matrix = np.vstack(msd_list)
    return window_dates, lags, msd_matrix

def plot_windowed_msd_heatmap(df, symbol, window_size=252, max_lag=30, demean=True):
    window_dates, lags, msd_matrix = compute_windowed_msd(
        df, window_size=window_size, max_lag=max_lag, demean=demean
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(
        msd_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[0, len(window_dates) - 1, lags[0], lags[-1]]
    )
    plt.colorbar(label="MSD")
    plt.ylabel("Lag (trading days)")
    plt.xlabel("Window index")
    plt.title(f"{symbol} - Windowed MSD of Log Price")
    plt.show()

def compute_local_msd_slope(df, window_size=252, fit_lags=10, demean=True):
    window_dates, lags, msd_matrix = compute_windowed_msd(
        df, window_size=window_size, max_lag=fit_lags, demean=demean
    )

    slopes = []
    for msd in msd_matrix:
        coeffs = np.polyfit(lags, msd, 1)
        slopes.append(coeffs[0])

    return window_dates, np.array(slopes)


def plot_local_msd_slope(df, symbol, window_size=252, fit_lags=10, demean=True):
    window_dates, slopes = compute_local_msd_slope(
        df, window_size=window_size, fit_lags=fit_lags, demean=demean
    )

    plt.figure(figsize=(12, 4))
    plt.plot(window_dates, slopes)
    plt.ylabel("Local MSD slope")
    plt.xlabel("Date")
    plt.title(f"{symbol} - Local MSD slope of Log Price")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    symbol = "A"
    category = "stocks"   # or "etfs"

    df = load_ticker(symbol, category)
    print_summary(df, symbol)
    plot_ticker(df, symbol, category)

    plot_windowed_msd_heatmap(df, symbol="A", window_size=252, max_lag=30, demean=True)
    plot_local_msd_slope(df, symbol="A", window_size=252, fit_lags=10, demean=True)