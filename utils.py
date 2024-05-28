import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from typing import List, Tuple

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


## scatter plot
def scatter(x: np.ndarray, 
         y: np.ndarray,
         alpha: np.ndarray,
         cmap: str = "binary",
         c_rev: str = "",
         size: int = 50,
         xlim: Tuple[float] = None, 
         ylim: Tuple[float] = None,
         xbin: float = None,
         ybin: float = None,
         xlabel: str = "x",
         ylabel: str = "y",
         xlog: bool = False, 
         ylog: bool = False,
         focus: List[np.ndarray] = None,
         fcolor: List[str] = "red",
         flabel: List[str] = None,
         save: bool = False,
         file_path: str = None,
         figsize: Tuple[int] = (6, 4),
         grid: bool = True,
         colorbar: bool = False,
        ) -> None:
    
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = figsize
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"
    p["ytick.minor.visible"] = False
    p["xtick.minor.visible"] = False
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    if grid:
        p["axes.grid"] = True
        p['axes.axisbelow'] = True # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xbin is not None:
        ax.set_xticks(np.linspace(x.min(), x.max(), int((x.max()-x.min())/xbin) + 1))
    if ybin is not None:
        ax.set_yticks(np.linspace(y.min(), y.max(), int((y.max()-y.min())/ybin) + 1))
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # contents
    cmap = plt.get_cmap(cmap + c_rev) # c_rev = "_r" for reverse
    vmin = alpha.min()*0.95 if alpha.min() > 0 else -1000
    sc = plt.scatter(x, y, c=alpha, cmap=cmap, s=size, zorder=1, vmin=vmin) #, vmax=np.percentile(alpha, 85)
    if focus is not None:
        # for f, color, label in zip(focus, fcolor, flabel):
            # plt.scatter(f[0], f[1], c=color, alpha=1., s=size, zorder=20, label=label)
        plt.scatter(focus[0], focus[1], c=fcolor, alpha=1., s=size, zorder=20, label=flabel)
    if flabel is not None:
        ax.legend()
    if colorbar:
        fig.colorbar(sc, ax=ax)
    
    if save:
        plt.savefig(file_path)
    else:
        plt.show()

## scatter plot
def scatter_categorical(
         x: np.ndarray, 
         y: np.ndarray,
         color: np.ndarray = None,
         size: int = 50,
         xlim: Tuple[float] = None, 
         ylim: Tuple[float] = None,
         xbin: float = None,
         ybin: float = None,
         xlabel: str = "x",
         ylabel: str = "y",
         xlog: bool = False, 
         ylog: bool = False,
         focus: List[np.ndarray] = None,
         fcolor: List[str] = "red",
         flabel: List[str] = None,
         save: bool = False,
         file_path: str = None,
         figsize: Tuple[int] = (6, 4),
         grid: bool = True,
         colorbar: bool = False,
        ) -> None:
    
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = figsize
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"
    p["ytick.minor.visible"] = False
    p["xtick.minor.visible"] = False
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    if grid:
        p["axes.grid"] = True
        p['axes.axisbelow'] = True # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xbin is not None:
        ax.set_xticks(np.linspace(x.min(), x.max(), int((x.max()-x.min())/xbin) + 1))
    if ybin is not None:
        ax.set_yticks(np.linspace(y.min(), y.max(), int((y.max()-y.min())/ybin) + 1))
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # contents
    sc = plt.scatter(x, y, c=color, s=size, zorder=1) #, vmax=np.percentile(alpha, 85)
    if focus is not None:
        for f, color, label in zip(focus, fcolor, flabel):
            plt.scatter(f[0], f[1], c=color, alpha=1., s=size, zorder=20, label=label)
    if flabel is not None:
        ax.legend()
    if colorbar:
        fig.colorbar(sc, ax=ax)
    
    if save:
        plt.savefig(file_path)
    else:
        plt.show()

## Box plot using pd.DataFrame
def boxplot(df: pd.DataFrame, 
        rotation: int = 0,
        marker: str = '^',
        markersize: float = 2,
        linewidth: float = 0.5,
        ylim: Tuple[float] = None,
        ybin: float = None,
        xlabel: str = "x",
        ylabel: str = "y", 
        save: bool = False,
        file_path: str = None,
        figsize: Tuple[int] = (6, 4),
        grid: bool = False,
        ):
    
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = figsize
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"
    p["ytick.minor.visible"] = False
    p["xtick.minor.visible"] = False
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    if grid:
        p["axes.grid"] = True
        p['axes.axisbelow'] = True # put grid behind
    
    # set figure
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    if ylim is not None: ax.set_ylim(ylim)
    if ybin is not None:
        ax.set_yticks(np.linspace(y.min(), y.max(), int((y.max()-y.min())/ybin) + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # contents
    boxprops=dict(linewidth=linewidth)
    whiskerprops=dict(linewidth=linewidth)
    medianprops = dict(linestyle='-', linewidth=1., color='black')
    flierprops=dict(marker=marker, markersize=markersize, linewidth=linewidth, markerfacecolor="black")
    ax.boxplot(df, labels=df.columns, 
               boxprops=boxprops, 
               whiskerprops=whiskerprops,
               medianprops=medianprops,
               flierprops=flierprops)
    plt.xticks(rotation = rotation)
    
    if save:
        plt.savefig(file_path)
    else:
        plt.show()

### Probability transition
def plot_df(df: pd.DataFrame,
        columns: np.ndarray,  
        name_columns: str = None,
        xlim: Tuple[float] = None, 
        ylim: Tuple[float] = None,
        xbin: float = None,
        ybin: float = None,
        xlabel: str = "x",
        ylabel: str = "y",
        xlog: bool = False, 
        ylog: bool = False,
        focus: int = None,
        fcolor: str = "red",
        flabel: str = None,
        save: bool = False,
        file_path: str = None,
        figsize: Tuple[int] = (6, 4),
        grid: bool = True,
        colorbar: bool = False,
        ) -> None:
    
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = figsize
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"
    p["ytick.minor.visible"] = False
    p["xtick.minor.visible"] = False
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    if grid:
        p["axes.grid"] = True
        p['axes.axisbelow'] = True # put grid behind
    
    # set figure
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    # if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    # if xbin is not None:
    #     ax.set_xticks(np.linspace(x.min(), x.max(), int((x.max()-x.min())/xbin) + 1))
    # if ybin is not None:
    #     ax.set_yticks(np.linspace(y.min(), y.max(), int((y.max()-y.min())/ybin) + 1))
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # contents
    # for i, idx_ in enumerate(df.index):
        # data = df.loc[idx_][columns]
    label = columns if name_columns is None else name_columns
    ax.plot(df[columns], label=label, zorder=1)
    
    # for emphasis
    if focus is not None:
        # plt.scatter(df.loc[focus][columns], c=fcolor, alpha=1., zorder=len(df)+1, label=flabel)
        plt.scatter(df[focus], c=fcolor, alpha=1., zorder=2, label=flabel)
    
    # show legend
    ax.legend()
    
    if save:
        plt.savefig(file_path)
    else:
        plt.show()

## describe stats of np.ndarray
def describe(a):
    return {
        "min": a.min(),
        "max": a.max(),
        "mean": a.mean(),
        "std": a.std(),
        "median": np.quantile(a, 0.5),
        "25ptile": np.quantile(a, 0.25),
        "75ptile": np.quantile(a, 0.75),
    }

## Visualize network
# %%
def plot_network(
        y: np.ndarray,
        v_st: np.ndarray,
        st_nodes: np.ndarray,
        node_data: pd.DataFrame,
        demand: pd.DataFrame = None,
        parking: pd.DataFrame = None,
        plot_node: bool = True,
        cmap: str = "binary",
        c_rev: str = "",
        size: int = 8,
        focus: Tuple[int] = (76, 53),
        fcolor: str = "red",
        save: bool = False,
        file_path: str = None,
        figsize: Tuple[int] = (6, 4),
        colorbar: bool = False,
        axis_off: bool = True,
        vmin_p: float = 0.05,
        vmax_p: float = None,
        v_width: float = 2,
        y_width: float = 3,
        f_width: float = 2,
        ):
        
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = figsize
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"
    p['axes.axisbelow'] = False # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    if axis_off: ax.set_axis_off()
    
    # node
    xs = node_data["x"].values
    ys = node_data["y"].values
    padding = ((xs.max() - xs.min())/20, (ys.max() - ys.min())/20)
    ax.set_xlim(xs.min()-padding[0], xs.max()+padding[0])
    ax.set_ylim(ys.min()-padding[1], ys.max()+padding[1])
    
    # contents
    # node plot
    cmap = plt.get_cmap(cmap + c_rev) # c_rev = "_r" for reverse
    if plot_node:
        ax.scatter(xs, ys, marker='o', color=cmap(1.0), s=size, zorder=100)
    
    for st_idx, v in enumerate(v_st):
        nodes = st_nodes[st_idx]
        i, j = nodes - 1
        max_val = v_st.max() if vmax_p is None else vmax_p
        c = max(v/max_val, vmin_p)
        ax.plot((xs[i], xs[j]), (ys[i], ys[j]), 
                marker=None, c=cmap(c), 
                linewidth=v_width, zorder=1)
        if y[st_idx] == 1:
            ax.plot((xs[i], xs[j]), (ys[i], ys[j]), marker=None, 
                    c="green", alpha=0.6, linewidth=y_width, zorder=5)
    
    if focus is not None:
        i, j = focus[0]-1, focus[1]-1
        ax.plot((xs[i], xs[j]), (ys[i], ys[j]), marker=None, c=fcolor, linewidth=f_width, zorder=1)
        
    if demand is not None:
        origins = demand["origin"].values
        dests = demand["destination"].values
        o_car = origins[(origins <= 273)]
        o_ped = origins[(origins > 273)] - 273
        d_car = dests[(dests <= 273)]
        d_ped = dests[(dests > 273)] - 273
        # car origins
        ax.scatter(xs[o_car-1], ys[o_car-1], marker="D", c="blue", alpha=0.9, s=30)
        # ped origins
        ax.scatter(xs[o_ped-1], ys[o_ped-1], marker="D", c="red", alpha=0.9, s=30)
        # car dests
        ax.scatter(xs[d_car-1], ys[d_car-1], marker="*", c="blue", alpha=0.9, s=50)
        # ped origins
        ax.scatter(xs[d_ped-1], ys[d_ped-1], marker="*", c="red", alpha=0.9, s=50)
    
    if parking is not None:
        cmap_p = plt.get_cmap("binary")
        from_ = parking["from_"].values - 1
        cap = parking["capacity"].values
        ax.scatter(xs[from_], ys[from_], color=cmap_p(cap/cap.max()), s=40, zorder=120)
        
    if save:
        plt.savefig(file_path)
    else:
        plt.show()

