import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ia.classes.preprocess import DaxProcesser
from ia.figure_tools.plot_segmentation import plot_segmentation


def test_load_segmentation_from_npy(tmp_path):
    segmentation = np.zeros((3, 8, 8), dtype=np.uint16)
    segmentation[0, 1:3, 1:4] = 1
    segmentation[1, 4:7, 4:7] = 2

    seg_path = tmp_path / "segmentation.npy"
    np.save(seg_path, segmentation)

    loaded = DaxProcesser._LoadSegmentation(str(seg_path), verbose=False)
    np.testing.assert_array_equal(loaded, segmentation)


def test_plot_segmentation_saves_projected_mask(tmp_path):
    segmentation = np.zeros((3, 8, 8), dtype=np.uint16)
    segmentation[0, 1:3, 1:4] = 1
    segmentation[2, 4:7, 4:7] = 2

    save_path = tmp_path / "segmentation.png"
    plot_segmentation(
        segmentation,
        save=True,
        save_filename=str(save_path),
        verbose=False,
    )

    assert save_path.exists()
