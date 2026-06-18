"""Append model-comparison and confusion-matrix cells to model_development.ipynb."""

import json
import pathlib

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "model_development.ipynb"

# ── NEW CELLS ──────────────────────────────────────────────────────────────────

md_comparison = {
    "cell_type": "markdown",
    "id": "a1b2c3d4",
    "metadata": {},
    "source": [
        "## 5. Model Comparison Table\n",
        "Which model performs best? We compare all three models on Accuracy, Precision, Recall, and F1-score.\n",
        "The **best value** in each metric column is highlighted in green."
    ]
}

code_comparison = {
    "cell_type": "code",
    "execution_count": None,
    "id": "b2c3d4e5",
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.metrics import accuracy_score, precision_recall_fscore_support\n",
        "import pandas as pd\n",
        "\n",
        "LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']\n",
        "\n",
        "# Models and their test-set predictions (produced in cells above)\n",
        "models_info = [\n",
        "    ('TF-IDF Bigram + SVM',          y_pred_svm),\n",
        "    ('Count Bigram + Logistic Reg.', y_pred_lr),\n",
        "    ('Count Bigram + Random Forest', y_pred_rf),\n",
        "]\n",
        "\n",
        "rows = []\n",
        "for name, y_pred in models_info:\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    prec, rec, f1, _ = precision_recall_fscore_support(\n",
        "        y_test, y_pred, average='weighted', zero_division=0)\n",
        "    rows.append({'Model': name, 'Accuracy': acc, 'Precision': prec,\n",
        "                 'Recall': rec, 'F1-Score': f1})\n",
        "\n",
        "comparison_df = (\n",
        "    pd.DataFrame(rows)\n",
        "    .sort_values('F1-Score', ascending=False)\n",
        "    .reset_index(drop=True)\n",
        ")\n",
        "comparison_df.index += 1\n",
        "\n",
        "medals = {1: '\\U0001f947 #1', 2: '\\U0001f948 #2', 3: '\\U0001f949 #3'}\n",
        "comparison_df.insert(0, 'Rank', [medals.get(i, f'#{i}') for i in comparison_df.index])\n",
        "comparison_df = comparison_df.reset_index(drop=True)\n",
        "\n",
        "metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
        "\n",
        "def highlight_max(s):\n",
        "    return ['background-color: #d4edda; font-weight: bold'\n",
        "            if v == s.max() else '' for v in s]\n",
        "\n",
        "styled = (\n",
        "    comparison_df.style\n",
        "    .apply(highlight_max, subset=metric_cols)\n",
        "    .format({col: '{:.4f}' for col in metric_cols})\n",
        "    .set_table_styles([\n",
        "        {'selector': 'thead th',\n",
        "         'props': [('background-color', '#343a40'), ('color', 'white'),\n",
        "                   ('font-size', '13px'), ('text-align', 'center'),\n",
        "                   ('padding', '8px 14px')]},\n",
        "        {'selector': 'td',\n",
        "         'props': [('text-align', 'center'), ('padding', '7px 14px'),\n",
        "                   ('font-size', '13px')]},\n",
        "        {'selector': 'tr:nth-child(even)',\n",
        "         'props': [('background-color', '#f8f9fa')]},\n",
        "        {'selector': 'table',\n",
        "         'props': [('border-collapse', 'collapse'), ('width', '100%')]},\n",
        "    ])\n",
        "    .set_caption('\\U0001f4ca Model Comparison \\u2014 Best values highlighted in green')\n",
        ")\n",
        "\n",
        "display(styled)\n",
        "best = comparison_df.iloc[0]\n",
        "print(f\"\\n\\u2705 Best model: {best['Model']}  |  F1-Score: {best['F1-Score']:.4f}\")\n"
    ]
}

md_confmat = {
    "cell_type": "markdown",
    "id": "c3d4e5f6",
    "metadata": {},
    "source": [
        "## 6. Confusion Matrix Visualization\n",
        "A confusion matrix shows which emotions are classified correctly (diagonal) and which ones\n",
        "are confused with each other (off-diagonal). Colors represent **row-normalised recall** per class;\n",
        "each cell also shows the raw count."
    ]
}

code_confmat = {
    "cell_type": "code",
    "execution_count": None,
    "id": "d4e5f6a7",
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import confusion_matrix\n",
        "import numpy as np\n",
        "\n",
        "LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']\n",
        "\n",
        "models_info = [\n",
        "    ('TF-IDF Bigram + SVM',          y_pred_svm),\n",
        "    ('Count Bigram + Logistic Reg.', y_pred_lr),\n",
        "    ('Count Bigram + Random Forest', y_pred_rf),\n",
        "]\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(22, 7))\n",
        "fig.suptitle('Confusion Matrices \\u2014 All Models',\n",
        "             fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "for ax, (name, y_pred) in zip(axes, models_info):\n",
        "    cm = confusion_matrix(y_test, y_pred, labels=LABEL_NAMES)\n",
        "    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)\n",
        "\n",
        "    # Annotation: raw count on top line, percentage on second line\n",
        "    annot = np.array(\n",
        "        [['{:d}\\n({:.0%})'.format(cm[i, j], cm_norm[i, j])\n",
        "          for j in range(len(LABEL_NAMES))]\n",
        "         for i in range(len(LABEL_NAMES))]\n",
        "    )\n",
        "\n",
        "    sns.heatmap(\n",
        "        cm_norm,\n",
        "        annot=annot,\n",
        "        fmt='',\n",
        "        cmap='Blues',\n",
        "        xticklabels=LABEL_NAMES,\n",
        "        yticklabels=LABEL_NAMES,\n",
        "        linewidths=0.5,\n",
        "        linecolor='#dee2e6',\n",
        "        vmin=0,\n",
        "        vmax=1,\n",
        "        ax=ax,\n",
        "        annot_kws={'size': 8},\n",
        "        cbar_kws={'shrink': 0.8},\n",
        "    )\n",
        "    ax.set_title(name, fontsize=11, fontweight='bold', pad=10)\n",
        "    ax.set_xlabel('Predicted Emotion', fontsize=10)\n",
        "    ax.set_ylabel('Actual Emotion', fontsize=10)\n",
        "    ax.tick_params(axis='x', rotation=30, labelsize=9)\n",
        "    ax.tick_params(axis='y', rotation=0, labelsize=9)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('\\u2705 Saved confusion_matrices.png')\n"
    ]
}

code_perclass = {
    "cell_type": "code",
    "execution_count": None,
    "id": "e5f6a7b8",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Per-class F1 bar chart for the best model (TF-IDF Bigram + SVM)\n",
        "from sklearn.metrics import classification_report\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']\n",
        "best_name, best_pred = 'TF-IDF Bigram + SVM', y_pred_svm\n",
        "\n",
        "report = classification_report(\n",
        "    y_test, best_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0)\n",
        "\n",
        "per_class = (\n",
        "    pd.DataFrame(report).T\n",
        "    .loc[LABEL_NAMES, ['precision', 'recall', 'f1-score', 'support']]\n",
        "    .rename(columns={'f1-score': 'f1'})\n",
        "    .astype({'support': int})\n",
        ")\n",
        "\n",
        "fig2, ax2 = plt.subplots(figsize=(9, 4))\n",
        "colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(LABEL_NAMES)))\n",
        "bars = ax2.barh(per_class.index, per_class['f1'],\n",
        "                color=colors, edgecolor='white', height=0.55)\n",
        "ax2.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)\n",
        "ax2.set_xlim(0, 1.10)\n",
        "ax2.set_xlabel('F1-Score', fontsize=11)\n",
        "ax2.set_title(f'Per-Class F1-Score \\u2014 {best_name}',\n",
        "              fontsize=12, fontweight='bold')\n",
        "ax2.axvline(per_class['f1'].mean(), color='tomato', linestyle='--',\n",
        "            linewidth=1.4,\n",
        "            label=f\"Mean F1 = {per_class['f1'].mean():.3f}\")\n",
        "ax2.legend(fontsize=9)\n",
        "ax2.spines[['top', 'right']].set_visible(False)\n",
        "ax2.invert_yaxis()\n",
        "plt.tight_layout()\n",
        "plt.savefig('per_class_f1.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f'Per-class breakdown for {best_name}:')\n",
        "display(\n",
        "    per_class.style\n",
        "    .format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1': '{:.3f}'})\n",
        "    .background_gradient(subset=['f1'], cmap='Blues')\n",
        ")\n"
    ]
}

# ── PATCH THE NOTEBOOK ─────────────────────────────────────────────────────────

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Guard: don't double-add if script is re-run
existing_ids = {cell.get("id") for cell in nb["cells"]}
new_cells = [md_comparison, code_comparison, md_confmat, code_confmat, code_perclass]
to_add = [c for c in new_cells if c["id"] not in existing_ids]

if not to_add:
    print("Nothing to add — cells already present.")
else:
    nb["cells"].extend(to_add)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK  Added {len(to_add)} cells. Notebook now has {len(nb['cells'])} cells total.")
