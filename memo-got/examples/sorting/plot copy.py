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
    methods_order=["default/sorting_64_2", "memoized/sorting_64_2", "default/sorting_64_4", 
                   "memoized/sorting_64_4", "default/sorting_64_10", "memoized/sorting_64_10"],
    model="GPT-4.1-mini",
    length=32,
    y_lower=0,
    cost_upper=0.0,
    display_solved=True,
    annotation_offset=0,
    display_left_ylabel=False,
    display_right_ylabel=False,
):
    methods = []
    for key in results:
        for tag in methods_order:
            if tag in key:
                methods.append(key)
    methods_order = methods
    # Extract scores based on the order
    scores_ordered = [
        [
            min(score, length)
            for score in results[method]["scores"]
            if score != 100 and score != 300
        ]
        for method in methods_order
    ]
    total_costs = [sum(results[method]["costs"]) for method in methods_order]
    llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])

    # Create figure and axis
    fig, ax = plt.subplots(dpi=150, figsize=(9, 5))

    # Create boxplots
    positions = range(1, len(methods_order) + 1)
    ax.boxplot(scores_ordered, positions=positions)
    # breakpoint()

    fig_fontsize = 12

    # Set the ticks and labels
    method_labels = ["GoT-Plain\n(2 unique\ndigits)", "GoT-Memo\n(2 unique\ndigits)", 
                     "GoT-Plain\n(4 unique\ndigits)",  "GoT-Memo\n(4 unique\ndigits)", 
                     "GoT-Plain\n(10 unique\ndigits)",  "GoT-Memo\n(10 unique\ndigits)"]
    plt.yticks(fontsize=fig_fontsize)
    ticks = range(1, len(method_labels) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(method_labels, fontsize=fig_fontsize)

    y_upper = length

    range_increase = 1
    if display_solved:
        if length < 48:
            range_increase = 2
        elif length < 96:
            range_increase = 4
        else:
            range_increase = 8

    ax.set_ylim(y_lower, y_upper + range_increase)
    ax1_yticks = range(
        y_lower, y_upper + 1, 2 if length < 48 else (4 if length < 96 else 8)
    )
    ax.set_yticks(ax1_yticks)
    if display_left_ylabel:
        ax.set_ylabel(f"Box Plots: Incorrectly sorted elements (lower is better)")

    ax.set_title(f"Sorting {length}-length lists with different unique elements")

    ax2 = ax.twinx()

    means = llm_calls.mean(axis=1)      # shape (2,)
    stds  = llm_calls.std(axis=1)       # shape (2,)
    print(means.shape)
    # ax2.bar(positions, total_costs, alpha=0.5, color="blue", label="Total Cost ($)")
    
    # Bar plot with mean + std
    ax2.bar(
        positions,
        means,
        alpha=0.5,
        color=["#4ddbff", "#ffae00", "#4ddbff", "#ffae00"],
        label="Avg LLM Calls",
        yerr=stds,
        capsize=4,
    )

    # Right axis styling
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)

    # Set upper limit based on raw llm_calls
    cost_upper = int(1.2 * np.max(llm_calls))
    ax2.set_ylim(0, cost_upper)

    # Let matplotlib auto-select reasonable ticks, then recolor them
    for tick in ax2.get_yticklabels():
        tick.set_color("#1919ff")

    if display_right_ylabel:
        ax2.set_ylabel(
            "Bar Plots: Average LLM Calls (lower is better)",
            color="#1919ff",
            fontsize=fig_fontsize,
        )

    if display_solved:
        annotation_height = y_upper + annotation_offset
        count = 1
        for method in methods_order:
            if method not in results:
                continue
            solved = results[method]["solved"]
            ax.text(
                count,
                annotation_height,
                f"{solved}",
                ha="center",
                va="bottom",
                fontsize=fig_fontsize,
            )
            count += 1

    model = model.replace(".", "").replace("-", "").lower()
    fig.savefig(f"sorting_{model}_{length}.pdf", bbox_inches="tight")


plot_results(
    get_plotting_data([sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]),
    length=int(sys.argv[7]),
    display_solved=False,
    display_left_ylabel=True,
    display_right_ylabel=True,
)