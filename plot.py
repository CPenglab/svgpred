import os
import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

def draw_genes(expr: pd.DataFrame,
               coor: pd.DataFrame,
               save_dir: str,
               spots_size = None,
               cmap = "inferno",
               alpha = 1.0,
               marker: str = 's',
               facecolor: str = "white",
               edgecolor: str = "face",
               linewidth: Union[int, float] = 0,
               dpi = 480):
    """
    Plotting spatial expression of multiple genes as a heatmap.

    ==========
    expr: pd.DataFrame.
        Spatial expression data with row names as barcodes and column names as gene names.
    coor: pd.DataFrame.
        Pixel coordinate data with row names as barcodes and containing 'x' and 'y' columns.
    spots_size: int.
        The size of spots to be plotted.
    cmap: str.
        Color map.
    alpha: float.
        The transparency of the spots to be plotted.
    marker: str.
        The shape of spots to be plotted.
    facecolor: str.
        The fill color of the image.
    edgecolor: str.
        The edge color of spots to be plotted.
    linewidth: int or float.
        The line width of spots to be plotted.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fig, ax = plt.subplots(nrows = 1, ncols = 1,
                            figsize = [5, 5])

    cmap = plt.cm.get_cmap(cmap)
    if isinstance(cmap, mpl.colors.ListedColormap):
        ax.set_prop_cycle(color=cmap.colors)

    if spots_size is None:
        spots_size = auto_spot_size(coor, ax)

    for i in tqdm(expr.columns):
        ax.scatter(coor.iloc[:, 0],
                   coor.iloc[:, 1],
                   c=expr.loc[:, i],
                   s=spots_size,
                   alpha=alpha,
                   marker = marker,
                   edgecolors = edgecolor,
                   linewidths = linewidth,
                   cmap=cmap)        

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(facecolor)

        fig.savefig(os.path.join(save_dir, f"{i}.jpg"),
                    dpi = dpi, bbox_inches="tight")
        
        ax.clear()

    plt.close(fig)

def auto_spot_size(coords, ax, scale=1.0):
    """
    coords: (n,2) spatial coordinates
    ax: matplotlib axis
    scale: scale factor of spots size
    """
    from sklearn.neighbors import NearestNeighbors
    coords = np.asarray(coords)
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    nn_dist = np.median(distances[:, 1])
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inch, height_inch = bbox.width, bbox.height
    x_scale = width_inch / (x_max - x_min)
    y_scale = height_inch / (y_max - y_min)
    scale_data_to_inch = min(x_scale, y_scale)
    diameter_points = nn_dist * scale_data_to_inch * 72 * scale
    s = diameter_points ** 2
    return s
