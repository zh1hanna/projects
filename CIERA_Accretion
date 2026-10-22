"""
Accretion luminosity of a black hole in an x-ray binary

Computes how bright a black hole is as it pulls in matter from a companion
star, using output from the COSMIC binary-evolution code, then checks whether
the system would be bright enough for an x-ray observatory to detect.

Physics (all from the CIERA REACH exercise):
    L_acc   = eta * Mdot_acc * c^2         mass-to-energy conversion
    eta     = 1 - sqrt(1 - (M_BH / (3 * M_BH0))^2),  capped at 0.42
    Mdot_BH = Mdot_acc * (1 - eta)         the hole keeps only what isn't radiated

Data and exercise: CIERA REACH program (Northwestern University),
https://github.com/CIERA-Northwestern/REACHpy
"""

import os
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/CIERA-Northwestern/REACHpy/main/"
    "Projects/AccretingBlackHoles/data/cosmic_data_1.csv"
)
DATA_PATH = os.path.join("data", "cosmic_data_1.csv")

# Constants in CGS units, so luminosity comes out in erg/s
C_CM_S = 2.998e10             # speed of light (cm/s)
M_SUN_G = 1.98847e33          # solar mass (g)
YEAR_S = 3.156e7              # seconds per year

ETA_MAX = 0.42                # efficiency of a maximally spinning black hole
BOLOMETRIC_CORRECTION = 0.8   # observed fraction of the true luminosity
OBSERVABLE_THRESHOLD = 1e37   # erg/s, typical Chandra detection limit


def load_data(path=DATA_PATH, url=DATA_URL):
    """Download the simulation CSV if needed and return it as a DataFrame."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)


def calc_eta(m_bh, m_bh0):
    """Mass-to-energy efficiency; grows as the black hole gains mass and spins up."""
    ratio = np.minimum(m_bh / (3 * m_bh0), 1.0)   # keeps the square root real
    eta = 1 - np.sqrt(1 - ratio ** 2)
    return np.minimum(eta, ETA_MAX)               # cannot spin faster than maximal


def calc_mdot_acc(mdot_bh, eta):
    """Rate matter arrives at the disk (g/s), given the rate the hole gains mass (Msun/yr)."""
    mdot_bh_cgs = mdot_bh * M_SUN_G / YEAR_S
    return mdot_bh_cgs / (1 - eta)


def calc_lacc(m_bh, m_bh0, mdot_bh):
    """
    Accretion luminosity in erg/s. Works on single values or whole arrays.

    m_bh    : current black hole mass (Msun)
    m_bh0   : black hole mass at formation (Msun)
    mdot_bh : rate the black hole gains mass (Msun/yr)
    """
    eta = calc_eta(m_bh, m_bh0)
    mdot_acc = calc_mdot_acc(mdot_bh, eta)
    return BOLOMETRIC_CORRECTION * eta * mdot_acc * C_CM_S ** 2


def lacc_with_loop(df):
    """Slow reference version: calls calc_lacc one row at a time."""
    result = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        result[i] = calc_lacc(row["mass_2"], row["mass0_2"], row["deltam_2"])
    return result


def plot_luminosity(time_myr, lacc, save_as="accretion_luminosity.png"):
    """Plot luminosity over time against the observability threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_myr, lacc, label="Accretion luminosity")
    ax.axhline(OBSERVABLE_THRESHOLD, color="purple", linestyle="--",
               label="Observable threshold (1e37 erg/s)")
    ax.set_yscale("log")
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Luminosity (erg/s)")
    ax.set_title("Accretion luminosity of the black hole over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_as, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Sanity check: M_BH = 11.645, M_BH0 = 11.611 Msun, Mdot = 3.026e-7 Msun/yr
    print(f"Test system L_acc: {calc_lacc(11.645, 11.611, 3.026e-7):.3e} erg/s")

    df = load_data()

    # Array method: one call, no loop (object 2 is the accreting black hole)
    start = time.perf_counter()
    lacc = calc_lacc(df["mass_2"].to_numpy(), df["mass0_2"].to_numpy(),
                     df["deltam_2"].to_numpy())
    array_time = time.perf_counter() - start

    # Row-by-row method, to confirm both agree and compare speed
    start = time.perf_counter()
    lacc_loop = lacc_with_loop(df)
    loop_time = time.perf_counter() - start

    print(f"Methods agree: {np.allclose(lacc, lacc_loop)}")
    print(f"Array method: {array_time:.5f} s | Loop method: {loop_time:.5f} s "
          f"({loop_time / array_time:.0f}x slower)")

    peak = lacc.max()
    print(f"Peak luminosity: {peak:.3e} erg/s")
    print(f"Observable: {peak > OBSERVABLE_THRESHOLD}")

    plot_luminosity(df["tphys"], lacc)
