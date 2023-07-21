import os
import pandas as pd
from typing import Union
import matplotlib as mpl
import matplotlib.pyplot as plt

def draw_genes(expr: pd.DataFrame,
               coor: pd.DataFrame,
               save_dir: str,
               spots_size = 3,
               cmap = "inferno",
               alpha = 1.0,
               marker: str = 'o',
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

    for i in expr.index:
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

