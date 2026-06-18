"""
1. Apply comparison table fix (DistilBERT included, inline HTML styles).
2. Reorder cells so that 6b (DistilBERT CM) sits right after 6 (classical CM),
   before the per-class F1 chart.
"""

import json
import pathlib

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "model_development.ipynb"

# ── Updated comparison table source (DistilBERT included) ─────────────────────
COMPARISON_SOURCE = [
    "from sklearn.metrics import accuracy_score, precision_recall_fscore_support\n",
    "from IPython.display import HTML\n",
    "import pandas as pd\n",
    "import pathlib\n",
    "\n",
    "LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']\n",
    "\n",
    "# Classical models: compute metrics from live predictions\n",
    "classical = [\n",
    "    ('TF-IDF Bigram + SVM',          y_pred_svm),\n",
    "    ('Count Bigram + Logistic Reg.', y_pred_lr),\n",
    "    ('Count Bigram + Random Forest', y_pred_rf),\n",
    "]\n",
    "rows = []\n",
    "for name, y_pred in classical:\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec, rec, f1, _ = precision_recall_fscore_support(\n",
    "        y_test, y_pred, average='weighted', zero_division=0)\n",
    "    rows.append({'Model': name, 'Accuracy': acc, 'Precision': prec,\n",
    "                 'Recall': rec, 'F1-Score': f1})\n",
    "\n",
    "# DistilBERT: load saved metrics from model_results.csv\n",
    "results_csv = pathlib.Path('..') / 'models' / 'classical' / 'model_results.csv'\n",
    "if results_csv.exists():\n",
    "    results_df = pd.read_csv(results_csv)\n",
    "    bert_row   = results_df[results_df['display_name'] == 'DistilBERT']\n",
    "    if not bert_row.empty:\n",
    "        b = bert_row.iloc[0]\n",
    "        rows.append({'Model': '\\U0001f916 DistilBERT (Transformer)',\n",
    "                     'Accuracy': b['accuracy'], 'Precision': b['precision'],\n",
    "                     'Recall':   b['recall'],   'F1-Score': b['f1']})\n",
    "    else:\n",
    "        print('\\u26a0\\ufe0f  DistilBERT not found in model_results.csv')\n",
    "else:\n",
    "    print('\\u26a0\\ufe0f  model_results.csv not found')\n",
    "\n",
    "comparison_df = (\n",
    "    pd.DataFrame(rows)\n",
    "    .sort_values('F1-Score', ascending=False)\n",
    "    .reset_index(drop=True)\n",
    ")\n",
    "\n",
    "medal_map   = {0: '\\U0001f947 #1', 1: '\\U0001f948 #2', 2: '\\U0001f949 #3',\n",
    "               3: '\\U0001f3c5 #4'}\n",
    "metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
    "best_vals   = {col: comparison_df[col].max() for col in metric_cols}\n",
    "\n",
    "TH = ('background-color:#1a1a2e; color:#ffffff; font-size:13px; '\n",
    "      'text-align:center; padding:10px 16px; '\n",
    "      'border-bottom:2px solid #f0a500; font-weight:bold;')\n",
    "TD_BASE = ('text-align:center; padding:9px 16px; font-size:13px; '\n",
    "           'border-bottom:1px solid #2a2a4a;')\n",
    "TD_ODD  = TD_BASE + ' color:#ffffff; background-color:#16213e;'\n",
    "TD_EVEN = TD_BASE + ' color:#ffffff; background-color:#0f3460;'\n",
    "TD_BERT = TD_BASE + ' color:#e8d5ff; background-color:#2d1b4e;'\n",
    "TD_GOLD = TD_BASE + ' color:#000000; font-weight:bold; background-color:#f0a500;'\n",
    "\n",
    "html = [\n",
    "    '<div style=\"font-family:sans-serif; margin:12px 0;\">',\n",
    "    '<p style=\"font-size:15px; font-weight:bold; color:#f0a500; margin-bottom:6px;\">'\n",
    "    '\\U0001f4ca Model Comparison \\u2014 Best values highlighted in gold</p>',\n",
    "    '<table style=\"border-collapse:collapse; width:100%;\">',\n",
    "    '<thead><tr>',\n",
    "    f'<th style=\"{TH}\">Rank</th>',\n",
    "    f'<th style=\"{TH}\">Model</th>',\n",
    "]\n",
    "for col in metric_cols:\n",
    "    html.append(f'<th style=\"{TH}\">{col}</th>')\n",
    "html.append('</tr></thead><tbody>')\n",
    "\n",
    "for i, row in comparison_df.iterrows():\n",
    "    is_bert = 'DistilBERT' in row['Model']\n",
    "    base    = TD_BERT if is_bert else (TD_ODD if i % 2 == 0 else TD_EVEN)\n",
    "    rank    = medal_map.get(i, f'#{i+1}')\n",
    "    html.append('<tr>')\n",
    "    html.append(f'<td style=\"{base}\">{rank}</td>')\n",
    "    html.append(f'<td style=\"{base}; text-align:left;\">{row[\"Model\"]}</td>')\n",
    "    for col in metric_cols:\n",
    "        val = row[col]\n",
    "        cs  = TD_GOLD if val == best_vals[col] else base\n",
    "        html.append(f'<td style=\"{cs}\">{val:.4f}</td>')\n",
    "    html.append('</tr>')\n",
    "\n",
    "html.append('</tbody></table>')\n",
    "html.append('<p style=\"font-size:11px; color:#888; margin-top:6px;\">'\n",
    "            '\\U0001f916 DistilBERT metrics loaded from model_results.csv (full dataset). '\n",
    "            'Classical model metrics computed live on the notebook test split.</p>')\n",
    "html.append('</div>')\n",
    "\n",
    "display(HTML(''.join(html)))\n",
    "best = comparison_df.iloc[0]\n",
    "print(f\"\\n\\u2705 Best model: {best['Model']}  |  F1-Score: {best['F1-Score']:.4f}\")\n",
]

# ── Load and patch notebook ────────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Patch comparison table
for cell in nb["cells"]:
    if cell.get("id") == "b2c3d4e5":
        cell["source"] = COMPARISON_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        print("OK  Patched comparison table (b2c3d4e5).")
        break

# ── Reorder cells ─────────────────────────────────────────────────────────────
# Target order of the LAST section:
#   d4e5f6a7  → classical CM code   (section 6)
#   f6a7b8c9  → DistilBERT CM md    (section 6b)  ← move here
#   f7a8b9c0  → DistilBERT CM code  (section 6b)  ← move here
#   e5f6a7b8  → per-class F1 code   (bonus)

ORDER = ["d4e5f6a7", "f6a7b8c9", "f7a8b9c0", "e5f6a7b8"]

# Build a lookup by id
cell_by_id = {c.get("id"): c for c in nb["cells"]}

# Check all four ids exist
missing = [oid for oid in ORDER if oid not in cell_by_id]
if missing:
    print(f"WARNING: cells not found: {missing}  — skipping reorder.")
else:
    # Remove the four cells from wherever they are
    nb["cells"] = [c for c in nb["cells"] if c.get("id") not in ORDER]

    # Find the position of the classical-CM markdown cell (c3d4e5f6) and insert after it
    insert_after_id = "c3d4e5f6"   # markdown "## 6. Confusion Matrix Visualization"
    insert_pos = None
    for idx, c in enumerate(nb["cells"]):
        if c.get("id") == insert_after_id:
            insert_pos = idx + 1
            break

    if insert_pos is None:
        # Fallback: just append at the end
        nb["cells"].extend([cell_by_id[oid] for oid in ORDER])
        print("WARNING: anchor cell not found — appended at end.")
    else:
        for offset, oid in enumerate(ORDER):
            nb["cells"].insert(insert_pos + offset, cell_by_id[oid])
        print(f"OK  Reordered 4 cells starting at position {insert_pos}.")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook saved. Total cells: {len(nb['cells'])}.")
