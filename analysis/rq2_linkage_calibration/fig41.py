"""Redraw Figure 4.1 with the approved figure functions (copied verbatim from code/build_final_thesis_figures_v1.py)."""
import ast, sys, numpy as np, pandas as pd, matplotlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]  # repository root (analysis/rq2_oat/20261012_v3_factorial/..)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SRC = str(REPO / 'code/build_final_thesis_figures_v1.py')
tree = ast.parse(open(SRC).read())
want = {'COLORS', 'CANVAS_INCHES', 'set_style', 'clean_axis', 'recurrence_rows', 'draw_interval_rows', 'figure_name_recurrence'}
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in want for t in n.targets))]
ns = {'np': np, 'pd': pd, 'plt': plt}
exec(compile(ast.Module(body=nodes, type_ignores=[]), SRC, 'exec'), ns)
rec = pd.read_csv(sys.argv[1]); out = sys.argv[2]
ns['set_style']()
fig = ns['figure_name_recurrence'](rec)
fig.savefig(out + '.png', format='png', dpi=500, facecolor='#FFFFFF', transparent=False)
fig.savefig(out + '.pdf', format='pdf', facecolor='#FFFFFF', transparent=False)
plt.close(fig)
from PIL import Image
im = Image.open(out + '.png'); print(im.size, im.mode)
if im.mode != 'RGB': im.convert('RGB').save(out + '.png', dpi=(500, 500))
