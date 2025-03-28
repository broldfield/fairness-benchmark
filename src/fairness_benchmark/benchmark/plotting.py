import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import os


def save_plot(args, type, name, fig):
    plot_location = "src/fairness_benchmark/data/plots/"
    name = f"{type}_{name}_{args.dataset}_{args.preprocess}_{args.sensitive}_{args.target}_{args.model}"
    path = plot_location + name + ".png"

    save_path = os.path.join(os.getcwd(), path)

    logger.info(f"Saving Plot to: {save_path}")
    fig.savefig(save_path)

    return


def plot_1(args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, disp_imp_arr):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr)
    ax1.set_xlabel("Classification Thresholds", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr)), color="r")
    ax2.set_ylabel("abs(1-disparate impact)", color="r", fontsize=16, fontweight="bold")
    ax2.axvline(best_class_thresh, color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def plot_2(
    args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, avg_odds_diff_arr
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr)
    ax1.set_xlabel("Classification Thresholds", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, avg_odds_diff_arr, color="r")
    ax2.set_ylabel("avg. odds diff.", color="r", fontsize=16, fontweight="bold")
    ax2.axvline(best_class_thresh, color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def create_plots(args, type, best_class_threshold, class_threshold, bal, disp, avg):
    fig_1 = plot_1(args, type, best_class_threshold, class_threshold, bal, disp)

    fig_2 = plot_2(args, type, best_class_threshold, class_threshold, bal, avg)

    save_plot(args, type, name="Plot_1", fig=fig_1)

    save_plot(args, type, name="Plot_2", fig=fig_2)
