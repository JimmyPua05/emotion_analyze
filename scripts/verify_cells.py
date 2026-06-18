import json, pathlib
nb = json.loads(pathlib.Path(
    "/home/jimmy_linux/anaconda_projects/nlp/Social media emotion analyzer project/notebooks/model_development.ipynb"
).read_text())
for i, c in enumerate(nb['cells']):
    src = c['source']
    preview = (src[0] if src else '')[:80].strip()
    print(f"Cell {i+1:2d} [{c['cell_type'][:4]}] | {preview}")
