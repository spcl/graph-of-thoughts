# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
# contributions: Robert Gerstenberger

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


# def get_complete_results(base_directory):
#     results_complete = {}
#     for folder_name in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, folder_name)
#         if os.path.isdir(folder_path):
#             results_complete[folder_name] = []
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".json"):
#                     file_path = os.path.join(folder_path, file_name)
#                     with open(file_path, "r") as f:
#                         data = json.load(f)
#                         results_complete[folder_name].append(
#                             {"key": int(file_name.split(".")[0]), "data": data}
#                         )
#         for key in results_complete.keys():
#             results_complete[key] = sorted(
#                 results_complete[key], key=lambda x: x["key"]
#             )
#     return results_complete


def get_complete_results(base_directories):
    results_complete = {}

    for base_directory in base_directories:
        for folder_name in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder_name)

            if os.path.isdir(folder_path):
                results_complete[folder_path] = []

                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            results_complete[folder_path].append(
                                {
                                    "key": int(file_name.split(".")[0]),
                                    "data": data,
                                }
                            )

                # sort entries for this folder
                results_complete[folder_path] = sorted(
                    results_complete[folder_path],
                    key=lambda x: x["key"],
                )

    return results_complete

def get_final_scores(results_complete):
    scores = {}
    for method in results_complete.keys():
        scores[method] = []
        for result in results_complete[method]:
            score = 100
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            llm_calls = 0
            for op in result["data"]:
                if "llm_calls" in op:
                    llm_calls = op["llm_calls"]
                if "operation" in op and op["operation"] == "ground_truth_evaluator":
                    try:
                        score = min(op["scores"])
                        solved = any(op["problem_solved"])
                    except:
                        continue
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
            scores[method].append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost, llm_calls]
            )
        scores[method] = sorted(scores[method], key=lambda x: x[0])
    return scores


def get_plotting_data(base_directory):
    results_complete = get_complete_results(base_directory)
    scores = get_final_scores(results_complete)
    results_plotting = {
        method: {
            "scores": [x[1] for x in scores[method]],
            "solved": sum([1 for x in scores[method] if x[2]]),
            "costs": [x[5] for x in scores[method]],
            "llm_calls": [x[6] for x in scores[method]],
        }
        for method in scores.keys()
    }
    return results_plotting


def plot_results(
    results,
    methods_order=["default/sorting_64_2", "memoized/sorting_64_2",
                   "default/sorting_64_4", "memoized/sorting_64_4",
                   "default/sorting_64_10", "memoized/sorting_64_10"],
    model="GPT-4.1-mini",
    length=32,
    y_lower=0,
    cost_upper=0.0,
    display_solved=True,
    annotation_offset=0,
    display_left_ylabel=False,
    display_right_ylabel=False,
):
    # --------------------------------------------------
    # Match methods
    # --------------------------------------------------
    methods = []
    for key in results:
        for tag in methods_order:
            if tag in key:
                methods.append(key)
    methods_order = methods

    # Scores
    scores_ordered = [
        [
            min(score, length)
            for score in results[method]["scores"]
            if score not in (100, 300)
        ]
        for method in methods_order
    ]

    llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])
    positions = range(1, len(methods_order) + 1)

    fig_fontsize = 12
    method_labels = [
        "GoT-Plain\n(2 unique\ndigits)", "GoT-Memo\n(2 unique\ndigits)",
        "GoT-Plain\n(4 unique\ndigits)", "GoT-Memo\n(4 unique\ndigits)",
        "GoT-Plain\n(10 unique\ndigits)", "GoT-Memo\n(10 unique\ndigits)",
    ]

    # --------------------------------------------------
    # Compute y-axis scaling (unchanged logic)
    # --------------------------------------------------
    y_upper = length
    range_increase = 1
    if display_solved:
        if length < 48:
            range_increase = 2
        elif length < 96:
            range_increase = 4
        else:
            range_increase = 8

    ax1_yticks = range(
        y_lower,
        y_upper + 1,
        2 if length < 48 else (4 if length < 96 else 8),
    )

    # ==================================================
    # FIGURE 1: BOX PLOTS
    # ==================================================
    fig1, ax1 = plt.subplots(dpi=150, figsize=(9, 5))

    ax1.boxplot(scores_ordered, positions=positions)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(method_labels, fontsize=fig_fontsize)
    ax1.set_ylim(y_lower, y_upper + range_increase)
    ax1.set_yticks(ax1_yticks)
    plt.yticks(fontsize=fig_fontsize)

    if display_left_ylabel:
        ax1.set_ylabel("Incorrectly sorted elements (lower is better)")

    ax1.set_title(f"Sorting {length}-length lists with different unique elements")

    # Solved annotations
    if display_solved:
        annotation_height = y_upper + annotation_offset
        for i, method in enumerate(methods_order, start=1):
            if method in results:
                solved = results[method]["solved"]
                ax1.text(
                    i,
                    annotation_height,
                    f"{solved}",
                    ha="center",
                    va="bottom",
                    fontsize=fig_fontsize,
                )

    model_tag = model.replace(".", "").replace("-", "").lower()
    fig1.savefig(f"sorting_box_{model_tag}_{length}.pdf", bbox_inches="tight")

    # ==================================================
    # FIGURE 2: BAR PLOTS (LLM calls)
    # ==================================================
    fig2, ax2 = plt.subplots(dpi=150, figsize=(9, 5))

    means = llm_calls.mean(axis=1)
    stds = llm_calls.std(axis=1)

    print(means)

    ax2.bar(
        positions,
        means,
        alpha=0.5,
        color=["#4ddbff", "#ffae00", "#4ddbff", "#ffae00"],
        yerr=stds,
        capsize=4,
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels(method_labels, fontsize=fig_fontsize)

    cost_upper = int(1.2 * np.max(llm_calls))
    ax2.set_ylim(0, cost_upper)

    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
    for tick in ax2.get_yticklabels():
        tick.set_color("#1919ff")

    if display_right_ylabel:
        ax2.set_ylabel(
            "Average LLM Calls (lower is better)",
            color="#1919ff",
            fontsize=fig_fontsize,
        )

    ax2.set_title(f"Sorting {length}-length lists with different unique elements")

    fig2.savefig(f"sorting_bar_{model_tag}_{length}.pdf", bbox_inches="tight")


plot_results(
    get_plotting_data([sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]),
    length=int(sys.argv[7]),
    display_solved=False,
    display_left_ylabel=True,
    display_right_ylabel=True,
)