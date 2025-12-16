# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
# contributions: Robert Gerstenberger, ChatGPT merge cleanup

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Read results from multiple result directories
# ---------------------------------------------------------
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

                results_complete[folder_path] = sorted(
                    results_complete[folder_path],
                    key=lambda x: x["key"],
                )

    return results_complete


# ---------------------------------------------------------
# 2. Extract (score, cost, solved, llm_calls)
# ---------------------------------------------------------
def get_final_scores(results_complete):
    scores = {}

    for method in results_complete.keys():
        method_scores = []

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
                        pass

                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]

            method_scores.append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost, llm_calls]
            )

        scores[method] = sorted(method_scores, key=lambda x: x[0])

    return scores


# ---------------------------------------------------------
# 3. Reduce to plotting-format
# ---------------------------------------------------------
def get_plotting_data(base_directories):
    results_complete = get_complete_results(base_directories)
    scores = get_final_scores(results_complete)

    return {
        method: {
            "scores": [x[1] for x in scores[method]],
            "solved": sum(1 for x in scores[method] if x[2]),
            "costs": [x[5] for x in scores[method]],
            "llm_calls": [x[6] for x in scores[method]],
        }
        for method in scores.keys()
    }


# ---------------------------------------------------------
# 4. PLOT (left = boxplot, right = mean±std llm-calls bar)
# ---------------------------------------------------------
# def plot_results(
#     results,
#     methods_order=["default/3", "memo/3", "default/6", "memo/6", "default/12", "memo/12"],
#     model="GPT-4.1-mini",
#     y_lower=0,
#     y_upper=40,
#     cost_upper=1.8,
#     display_solved=True,
#     annotation_offset=1,
#     display_left_ylabel=False,
#     display_right_ylabel=False,
# ):

#     # match folder paths containing keywords
#     methods = []
#     for key in results:
#         for tag in methods_order:
#             if tag in key:
#                 methods.append(key)
#     methods_order = methods

#     # Filter scores
#     scores_ordered = [
#         [score for score in results[method]["scores"] if score != 100 and score != 300]
#         for method in methods_order
#     ]

#     # Collect LLM calls
#     llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])

#     # Create figure
#     fig, ax = plt.subplots(dpi=150, figsize=(9, 5))
#     positions = range(1, len(methods_order) + 1)

#     # Left boxplot
#     ax.boxplot(scores_ordered, positions=positions)

#     fig_fontsize = 12

#     # X labels
#     ax.set_xticks(positions)
#     ax.set_xticklabels(["GoT-Plain\n(3 unique\nsentences)", "GoT-Memo\n(3 unique\nsentences)", 
#                         "GoT-Plain\n(6 unique\nsentences)", "GoT-Memo\n(6 unique\nsentences)", 
#                         "GoT-Plain\n(12 unique\nsentences)", "GoT-Memo\n(12 unique\nsentences)"], fontsize=fig_fontsize)

#     ax.set_ylim(y_lower, (y_upper + 2) if display_solved else y_upper + 1)
#     plt.yticks(fontsize=fig_fontsize)
#     if display_left_ylabel:
#         ax.set_ylabel(f"Box Plots: Missed keywords (lower is better)", fontsize=fig_fontsize)

#     ax.set_title(f"Keyword Counting with different sets of sentences")

#     # ---------------------------------------------------------
#     # RIGHT AXIS (LLM calls: bar with mean ± std)
#     # ---------------------------------------------------------
#     ax2 = ax.twinx()

#     means = llm_calls.mean(axis=1)
#     stds = llm_calls.std(axis=1)

#     ax2.bar(
#         positions,
#         means,
#         alpha=0.5,
#         color=["#4ddbff", "#ffae00", "#4ddbff", "#ffae00", "#4ddbff", "#ffae00"],
#         label="Avg LLM Calls",
#         yerr=stds,
#         capsize=4,
#     )

#     # Formatting
#     ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)

#     # Automatic reasonable limit
#     cost_upper = int(1.2 * np.max(llm_calls))
#     ax2.set_ylim(0, cost_upper)

#     # Recolor ticks
#     for tick in ax2.get_yticklabels():
#         tick.set_color("#1919ff")

#     if display_right_ylabel:
#         ax2.set_ylabel("Bar Plots: Average LLM Calls (lower is better)", color="#1919ff", fontsize=fig_fontsize)

#     # Solved counts annotation
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
#                 f"{solved}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=fig_fontsize,
#             )
#             count += 1

#     # Save
#     model_tag = model.replace(".", "").replace("-", "").lower()
#     fig.savefig(f"keyword_counting_{model_tag}.pdf", bbox_inches="tight")


def plot_results(
    results,
    methods_order=["default/3", "memo/3", "default/6", "memo/6", "default/12", "memo/12"],
    model="GPT-4.1-mini",
    y_lower=0,
    y_upper=40,
    cost_upper=1.8,
    display_solved=True,
    annotation_offset=1,
    display_left_ylabel=False,
    display_right_ylabel=False,
):

    # -----------------------------
    # Extract matching methods
    # -----------------------------
    methods = []
    for key in results:
        for tag in methods_order:
            if tag in key:
                methods.append(key)
    methods_order = methods

    # Filter scores
    scores_ordered = [
        [score for score in results[method]["scores"] if score not in (100, 300)]
        for method in methods_order
    ]

    # LLM calls
    llm_calls = np.asarray([results[method]["llm_calls"] for method in methods_order])

    positions = range(1, len(methods_order) + 1)
    fig_fontsize = 12

    xtick_labels = [
        "GoT-Plain\n(3 unique\nsentences)",
        "GoT-Memo\n(3 unique\nsentences)",
        "GoT-Plain\n(6 unique\nsentences)",
        "GoT-Memo\n(6 unique\nsentences)",
        "GoT-Plain\n(12 unique\nsentences)",
        "GoT-Memo\n(12 unique\nsentences)",
    ]

    # =====================================================================
    #  FIGURE 1: BOX PLOT (missed keywords)
    # =====================================================================
    fig1, ax1 = plt.subplots(dpi=150, figsize=(9, 5))

    ax1.boxplot(scores_ordered, positions=positions)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(xtick_labels, fontsize=fig_fontsize)

    ax1.set_ylim(y_lower, (y_upper + 2) if display_solved else y_upper + 1)
    plt.yticks(fontsize=fig_fontsize)

    if display_left_ylabel:
        ax1.set_ylabel("Missed keywords (lower is better)", fontsize=fig_fontsize)

    ax1.set_title("Keyword Counting with different sets of sentences")

    # Annotation for "solved"
    if display_solved:
        annotation_height = y_upper + annotation_offset
        count = 1
        for method in methods_order:
            if method in results:
                solved = results[method]["solved"]
                ax1.text(
                    count,
                    annotation_height,
                    f"{solved}",
                    ha="center",
                    va="bottom",
                    fontsize=fig_fontsize,
                )
            count += 1

    model_tag = model.replace(".", "").replace("-", "").lower()
    fig1.savefig(f"keyword_counting_box_{model_tag}.pdf", bbox_inches="tight")

    # =====================================================================
    #  FIGURE 2: BAR PLOT (LLM calls mean ± std)
    # =====================================================================
    fig2, ax2 = plt.subplots(dpi=150, figsize=(9, 5))

    means = llm_calls.mean(axis=1)
    stds = llm_calls.std(axis=1)

    print(means)

    ax2.bar(
        positions,
        means,
        alpha=0.5,
        color=["#4ddbff", "#ffae00", "#4ddbff", "#ffae00", "#4ddbff", "#ffae00"],
        yerr=stds,
        capsize=4,
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels(xtick_labels, fontsize=fig_fontsize)

    cost_upper = int(1.2 * np.max(llm_calls))
    ax2.set_ylim(0, cost_upper)

    # Styling for right axis
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
    for tick in ax2.get_yticklabels():
        tick.set_color("#1919ff")

    if display_right_ylabel:
        ax2.set_ylabel("Average LLM Calls (lower is better)", fontsize=fig_fontsize, color="#1919ff")

    ax2.set_title("Keyword Counting with different sets of sentences")

    fig2.savefig(f"keyword_counting_bar_{model_tag}.pdf", bbox_inches="tight")


# ---------------------------------------------------------
# 5. Run
# ---------------------------------------------------------
plot_results(
    get_plotting_data([sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]),
    display_solved=False,
    annotation_offset=-0.3,
    y_upper=35,
    display_left_ylabel=True,
    display_right_ylabel=True,
    cost_upper=9,
)
