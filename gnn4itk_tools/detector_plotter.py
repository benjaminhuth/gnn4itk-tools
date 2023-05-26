import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DetectorPlotter:
    def __init__(self, detector_path, r_max=np.inf, abs_z_max=np.inf):
        self.detector = pd.read_csv(detector_path)
        self.detector["cr"] = np.hypot(self.detector.cx, self.detector.cy)
        self.detector = self.detector[abs(self.detector.cz) < abs_z_max].copy()
        self.detector = self.detector[self.detector.cr < r_max].copy()

    def get_fig_ax(self, figsize=None, s=5, color="lightgrey", alpha=1.0):
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].scatter(self.detector.cz, self.detector.cr, s=s, color=color, alpha=1.0)
        ax[0].set_xlabel("z [mm]")
        ax[0].set_ylabel("r [mm]")

        ax[1].scatter(self.detector.cx, self.detector.cy, s=s, color=color, alpha=1.0)
        ax[1].set_xlabel("x [mm]")
        ax[1].set_ylabel("y [mm]")

        return fig, ax
