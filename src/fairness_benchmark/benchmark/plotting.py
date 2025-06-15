import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import os

from fairness_benchmark.utils.loading import save_plot


def plot_disp(
    args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, disp_imp_arr
):
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


def plot_avg_odd(
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


def plot_stat_parity(
    args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, stat_parity_arr
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr)
    ax1.set_xlabel("Classification Thresholds", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, stat_parity_arr, color="r")
    ax2.set_ylabel("Stat Parity Diff.", color="r", fontsize=16, fontweight="bold")
    ax2.axvline(best_class_thresh, color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def plot_eq_opp(
    args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, eq_opp_arr
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr)
    ax1.set_xlabel("Classification Thresholds", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, eq_opp_arr, color="r")
    ax2.set_ylabel("Equal Opp.", color="r", fontsize=16, fontweight="bold")
    ax2.axvline(best_class_thresh, color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def plot_theil_index(
    args, type, best_class_thresh, class_thresh_arr, bal_acc_arr, theil_index_arr
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr)
    ax1.set_xlabel("Classification Thresholds", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, theil_index_arr, color="r")
    ax2.set_ylabel("Theil Index", color="r", fontsize=16, fontweight="bold")
    ax2.axvline(best_class_thresh, color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def create_plots(args, type, best_class_threshold, class_threshold, metric_df):
    fig_disp = plot_disp(
        args,
        type,
        best_class_threshold,
        class_threshold,
        metric_df["Balanced Average"],
        metric_df["Disparate Impact"],
    )
    save_plot(args, type, name="Plot_Disp", fig=fig_disp)
    fig_avg_odd = plot_avg_odd(
        args,
        type,
        best_class_threshold,
        class_threshold,
        metric_df["Balanced Average"],
        metric_df["Average Odds Difference"],
    )

    save_plot(args, type, name="Plot_Avg_Odd", fig=fig_avg_odd)
    fig_stat_parity = plot_stat_parity(
        args,
        type,
        best_class_threshold,
        class_threshold,
        metric_df["Balanced Average"],
        metric_df["Statistical Parity Difference"],
    )
    save_plot(args, type, name="Plot_Stat_Parity", fig=fig_stat_parity)
    fig_eq_opp = plot_eq_opp(
        args,
        type,
        best_class_threshold,
        class_threshold,
        metric_df["Balanced Average"],
        metric_df["Equal Opportunity Difference"],
    )
    save_plot(args, type, name="Plot_Eq_Opp", fig=fig_eq_opp)
    fig_theil_index = plot_theil_index(
        args,
        type,
        best_class_threshold,
        class_threshold,
        metric_df["Balanced Average"],
        metric_df["Theil Index"],
    )
    save_plot(args, type, name="Plot_Theil_Index", fig=fig_theil_index)

    plt.close("all")
