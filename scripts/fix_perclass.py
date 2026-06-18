"""Patch per-class F1 cell to show all 4 models grouped side by side."""

import json
import pathlib

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "model_development.ipynb"

NEW_SOURCE = [
    "# Per-class F1 comparison — all models side by side\n",
    "from sklearn.metrics import classification_report\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import pathlib\n",
    "\n",
    "LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']\n",
    "\n",
    "# ── Classical models (live predictions) ──────────────────────────────────────\n",
    "classical_models = [\n",
    "    ('TF-IDF Bigram + SVM',          y_pred_svm),\n",
    "    ('Count Bigram + Logistic Reg.', y_pred_lr),\n",
    "    ('Count Bigram + Random Forest', y_pred_rf),\n",
    "]\n",
    "\n",
    "all_models = []   # list of (name, f1_per_class_array)\n",
    "for name, y_pred in classical_models:\n",
    "    report = classification_report(\n",
    "        y_test, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0)\n",
    "    f1s = [report[label]['f1-score'] for label in LABEL_NAMES]\n",
    "    all_models.append((name, np.array(f1s)))\n",
    "\n",
    "# ── DistilBERT: load per-class F1 from model_results_details.json ─────────────\n",
    "details_path = pathlib.Path('..') / 'models' / 'classical' / 'model_results_details.json'\n",
    "if details_path.exists():\n",
    "    import json as _json\n",
    "    details = _json.loads(details_path.read_text(encoding='utf-8'))\n",
    "    bert_detail = next((d for d in details if d.get('display_name') == 'DistilBERT'), None)\n",
    "    if bert_detail and 'classification_report' in bert_detail:\n",
    "        cr = bert_detail['classification_report']\n",
    "        f1s = [cr.get(label, {}).get('f1-score', 0.0) for label in LABEL_NAMES]\n",
    "        all_models.append(('DistilBERT', np.array(f1s)))\n",
    "    else:\n",
    "        print('\\u26a0\\ufe0f  DistilBERT per-class report not found in details JSON.')\n",
    "\n",
    "# ── Grouped bar chart ─────────────────────────────────────────────────────────\n",
    "n_models  = len(all_models)\n",
    "n_labels  = len(LABEL_NAMES)\n",
    "x         = np.arange(n_labels)\n",
    "bar_width = 0.18\n",
    "colors    = ['#4e9af1', '#f0a500', '#2ecc71', '#9b59b6']   # blue, gold, green, purple\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(13, 5))\n",
    "\n",
    "for idx, (name, f1s) in enumerate(all_models):\n",
    "    offsets = x + (idx - (n_models - 1) / 2) * bar_width\n",
    "    bars = ax.bar(offsets, f1s, width=bar_width,\n",
    "                  label=name, color=colors[idx % len(colors)],\n",
    "                  edgecolor='white', linewidth=0.6)\n",
    "    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)\n",
    "\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(LABEL_NAMES, fontsize=11)\n",
    "ax.set_ylim(0, 1.12)\n",
    "ax.set_ylabel('F1-Score', fontsize=11)\n",
    "ax.set_xlabel('Emotion Class', fontsize=11)\n",
    "ax.set_title('Per-Class F1-Score \\u2014 All Models Compared', fontsize=13, fontweight='bold')\n",
    "ax.legend(fontsize=9, loc='lower right')\n",
    "ax.spines[['top', 'right']].set_visible(False)\n",
    "ax.yaxis.grid(True, linestyle='--', alpha=0.4)\n",
    "ax.set_axisbelow(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('per_class_f1_all_models.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('\\u2705 Saved per_class_f1_all_models.png')\n",
    "\n",
    "# ── Summary table ─────────────────────────────────────────────────────────────\n",
    "table = pd.DataFrame(\n",
    "    {name: f1s for name, f1s in all_models},\n",
    "    index=LABEL_NAMES\n",
    ").round(3)\n",
    "\n",
    "display(\n",
    "    table.style\n",
    "    .background_gradient(cmap='Blues', axis=None, vmin=0.6, vmax=1.0)\n",
    "    .format('{:.3f}')\n",
    "    .set_caption('Per-class F1-Score for all models (higher = better)')\n",
    ")\n",
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("id") == "e5f6a7b8":
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        print("OK  Updated per-class F1 cell (e5f6a7b8) to show all models.")
        break
else:
    print("ERROR: cell e5f6a7b8 not found.")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook saved.")
