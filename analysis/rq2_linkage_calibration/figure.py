import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
df = pd.read_csv('factorial_runs.csv')
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
T = {
 'zh': dict(font='Noto Sans CJK SC', x='地址完全相同时所需的最低名称相似度（名称下限）', y='筛选后的配对数',
            s1='门牌号锁定成立', s2='门牌号锁定不成立', prim='主设定', title='3,087种参数组合下的配对数'),
 'en': dict(font='DejaVu Sans', x='Minimum name similarity needed at an identical address (name floor)', y='Links after eligibility filtering',
            s1='House-number lock holds', s2='House-number lock fails', prim='Primary setting', title='Links across 3,087 parameter combinations'),
}
for lang, t in T.items():
    plt.rcParams.update({'font.family': t['font'], 'font.size': 9, 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=300)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    a = df[df.lock]; b = df[~df.lock]
    ax.scatter(b.name_floor, b.post_links, s=18, marker='^', c=ORANGE, alpha=0.35, linewidths=0, label=t['s2'], zorder=2)
    ax.scatter(a.name_floor, a.post_links, s=16, marker='o', c=BLUE, alpha=0.35, linewidths=0, label=t['s1'], zorder=3)
    ax.scatter([55], [150], s=90, facecolors='none', edgecolors=INK, linewidths=1.4, zorder=5)
    ax.annotate(f"{t['prim']} (0.8, 60, 0.85, 75)", xy=(55, 150), xytext=(6, 134), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8), zorder=6)
    ax.set_xlabel(t['x'], color=INK2); ax.set_ylabel(t['y'], color=INK2)
    ax.set_title(t['title'], color=INK, fontsize=10, loc='left')
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlim(-3, 92); ax.set_ylim(122, 168)
    leg = ax.legend(loc='upper right', frameon=False, labelcolor=INK)
    for h in leg.legend_handles: h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(f'figure_A5_6_namefloor_{lang}.png', facecolor=SURF)
    plt.close(fig)
print('ok')
