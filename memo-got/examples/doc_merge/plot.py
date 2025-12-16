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
            score = 0
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in reversed(result["data"]):
                if "llm_calls" in op:
                    llm_calls = op["llm_calls"]
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
                if "operation" in op and op["operation"] == "score":
                    try:
                        score = max(op["scores"])
                        break
                    except:
                        continue
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


# def plot_results(
#     results,
#     display_solved,
#     methods_order=["plain-got", "alpha_0.9", "alpha_0.5"],
#     model="GPT-4.1-mini",
#     num_ndas=16,
#     y_lower=0,
#     y_upper=10,
#     cost_upper=1.8,
#     annotation_offset=1,
#     display_left_ylabel=False,
#     display_right_ylabel=False,
# ):
#     methods = []
#     for key in results:
#         for tag in methods_order:
#             if tag in key:
#                 methods.append(key)
#     methods_order = methods
#     # Extract scores based on the order
#     scores_ordered = [
#         [score for score in results[method]["scores"]] for method in methods_order
#     ]

#     total_costs = [sum(results[method]["costs"]) for method in methods_order]
#     llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])

#     # Create figure and axis
#     fig, ax = plt.subplots(dpi=150, figsize=(5, 5))

#     # Create boxplots
#     positions = range(1, len(methods_order) + 1)
#     ax.boxplot(scores_ordered, positions=positions)

#     fig_fontsize = 12

#     # Set the ticks and labels
#     method_labels = ["GoT-Plain\n(Alpha=1.0)", "GoT-Memo\n(Alpha=0.9)", "GoT-Memo\n(Alpha=0.5)"]
#     plt.yticks(fontsize=fig_fontsize)
#     ax.set_xticks(range(1, len(methods_order) + 1))
#     ax.set_xticks(range(1, len(methods_order) + 1))
#     ax.set_xticklabels(method_labels, fontsize=fig_fontsize)

#     ax.set_ylim(y_lower, 12)
#     plt.yticks(fontsize=fig_fontsize)

#     if display_left_ylabel:
#         ax.set_ylabel(
#             f"Box Plots: Score (out of 10, higher is better)", fontsize=fig_fontsize
#         )

#     ax.set_title(f"Document Merging under different Thresholds")

#     ax2 = ax.twinx()

#     means = llm_calls.mean(axis=1)      # shape (2,)
#     stds  = llm_calls.std(axis=1)       # shape (2,)
#     # ax2.bar(positions, total_costs, alpha=0.5, color="blue", label="Total Cost ($)")
    
#     # Bar plot with mean + std
#     ax2.bar(
#         positions,
#         means,
#         alpha=0.5,
#         color=["#4ddbff", "#ffae00","#ffae00"],
#         label="Avg LLM Calls",
#         yerr=stds,
#         capsize=4,
#     )

#     # Right axis styling
#     ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)

#     # Set upper limit based on raw llm_calls
#     cost_upper = int(1.2 * np.max(llm_calls))
#     ax2.set_ylim(0, cost_upper)

#     # Let matplotlib auto-select reasonable ticks, then recolor them
#     for tick in ax2.get_yticklabels():
#         tick.set_color("#1919ff")

#     if display_right_ylabel:
#         ax2.set_ylabel(
#             "Bar Plots: Average LLM Calls (lower is better)",
#             color="#1919ff",
#             fontsize=fig_fontsize,
#         )

#     if display_solved:
#         annotation_height = y_upper + annotation_offset
#         count = 1
#         for method in methods_order:
#             if method not in results:
#                 continue
#             solved = results[method]["solved"]
#             ax.text(
#                 count,
#                 annotation_height,
#                 f"Solved: {solved}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=fig_fontsize,
#             )
#             count += 1

#     model = model.replace(".", "").replace("-", "").lower()
#     fig.savefig(f"doc_merge_{model}_{num_ndas}.pdf", bbox_inches="tight")


def plot_results(
    results,
    display_solved,
    methods_order=["plain-got", "alpha_0.9", "alpha_0.5"],
    model="GPT-4.1-mini",
    num_ndas=16,
    y_lower=0,
    y_upper=10,
    cost_upper=1.8,
    annotation_offset=1,
    display_left_ylabel=False,
    display_right_ylabel=False,
):
    # -----------------------------
    # Extract methods and data
    # -----------------------------
    methods = []
    for key in results:
        for tag in methods_order:
            if tag in key:
                methods.append(key)
    methods_order = methods

    scores_ordered = [
        [score for score in results[method]["scores"]] 
        for method in methods_order
    ]

    total_costs = [sum(results[method]["costs"]) for method in methods_order]
    llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])

    positions = range(1, len(methods_order) + 1)
    method_labels = ["GoT-Plain\n(Alpha=1.0)", "GoT-Memo\n(Alpha=0.9)", "GoT-Memo\n(Alpha=0.5)"]
    fig_fontsize = 12

    # =====================================================================
    #  FIGURE 1: BOX PLOT (Scores)
    # =====================================================================
    fig1, ax1 = plt.subplots(dpi=150, figsize=(5, 5))

    ax1.boxplot(scores_ordered, positions=positions)

    ax1.set_xticks(range(1, len(methods_order) + 1))
    ax1.set_xticklabels(method_labels, fontsize=fig_fontsize)
    ax1.set_ylim(y_lower, 12)
    plt.yticks(fontsize=fig_fontsize)

    if display_left_ylabel:
        ax1.set_ylabel(
            "Score (out of 10, higher is better)",
            fontsize=fig_fontsize
        )

    ax1.set_title("Document Merging under different Thresholds")

    # Annotate “solved”
    if display_solved:
        annotation_height = y_upper + annotation_offset
        count = 1
        for method in methods_order:
            if method not in results:
                continue
            solved = results[method]["solved"]
            ax1.text(
                count, annotation_height,
                f"Solved: {solved}",
                ha="center", va="bottom",
                fontsize=fig_fontsize,
            )
            count += 1

    # Save box plot
    model_tag = model.replace(".", "").replace("-", "").lower()
    fig1.savefig(f"doc_merge_box_{model_tag}_{num_ndas}.pdf", bbox_inches="tight")

    # =====================================================================
    #  FIGURE 2: BAR PLOT (LLM Calls)
    # =====================================================================
    fig2, ax2 = plt.subplots(dpi=150, figsize=(5, 5))

    means = llm_calls.mean(axis=1)
    stds = llm_calls.std(axis=1)

    ax2.bar(
        positions,
        means,
        alpha=0.5,
        color=["#4ddbff", "#ffae00", "#ffae00"],
        label="Avg LLM Calls",
        yerr=stds,
        capsize=4,
    )

    cost_upper = int(1.2 * np.max(llm_calls))
    ax2.set_ylim(0, cost_upper)

    ax2.set_xticks(range(1, len(methods_order) + 1))
    ax2.set_xticklabels(method_labels, fontsize=fig_fontsize)

    # Right-axis text color styling
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
    for tick in ax2.get_yticklabels():
        tick.set_color("#1919ff")

    if display_right_ylabel:
        ax2.set_ylabel(
            "Average LLM Calls (lower is better)",
            color="#1919ff",
            fontsize=fig_fontsize,
        )

    ax2.set_title("Document Merging under different Thresholds")
    fig2.savefig(f"doc_merge_bar_{model_tag}_{num_ndas}.pdf", bbox_inches="tight")


plot_results(
    get_plotting_data([sys.argv[1], sys.argv[2], sys.argv[3]]),
    display_solved=False,
    display_left_ylabel=True,
    display_right_ylabel=True,
)