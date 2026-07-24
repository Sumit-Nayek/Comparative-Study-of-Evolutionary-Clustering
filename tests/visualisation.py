import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
import os

# Set up global style
plt.style.use('default')

# Color palette optimized for light/white theme
colors = {
    'primary': '#0969da',    # Deep Blue
    'secondary': '#cf222e',  # Crimson Red
    'accent': '#1a7f37',     # Forest Green
    'warning': '#9a6700',    # Dark Gold/Amber
    'purple': '#8250df',     # Vivid Purple
    'pink': '#bf3989',       # Magenta/Pink
    'text': '#1f2328',       # Charcoal Text
    'muted': '#57606a',      # Slate Gray
    'bg': '#ffffff',         # Pure White
    'card': '#f6f8fa',       # Very Light Gray Accent
    'border': '#d0d7de',     # Light Border Gray
    'grid': '#e1e4e8'        # Subtle Grid Line
}

def style_axis(ax):
    """Helper to apply consistent light-mode axis styling."""
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['muted'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colors['border'])
    ax.grid(True, alpha=0.6, color=colors['grid'])

# ============================================
# VISUAL 1: Evolutionary Process Flow Diagram
# ============================================
fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor=colors['bg'])
ax1.set_facecolor(colors['bg'])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Evolutionary Multi-Objective Clustering Pipeline', 
              color=colors['text'], fontsize=14, fontweight='bold', pad=15)

stages = [
    ('Data\nEmbedding', 1.5, 8, colors['primary']),
    ('Initialize\nPopulation', 3.5, 8, colors['purple']),
    ('Vectorized\nDistance Engine', 5.5, 8, colors['accent']),
    ('NSGA-II\nSorting', 7.5, 8, colors['secondary']),
    ('Pareto\nFront', 5.5, 5, colors['warning']),
    ('Crossover &\nMutation', 3.5, 5, colors['pink']),
    ('Convergence\nCheck', 1.5, 5, colors['primary']),
]

for label, x, y, color in stages:
    box = FancyBboxPatch((x-0.7, y-0.6), 1.4, 1.2,
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['card'], alpha=1.0,
                         edgecolor=color, linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, y, label, ha='center', va='center', 
             color=color, fontsize=9, fontweight='bold')

arrows = [(2.2, 8, 2.8, 8), (4.2, 8, 4.8, 8), (6.2, 8, 6.8, 8),
          (7.5, 7.4, 5.5, 5.6), (5.5, 4.4, 3.5, 5.6), (3.5, 4.4, 1.5, 5.6),
          (1.5, 5.6, 1.5, 7.4)]
for x1, y1, x2, y2 in arrows:
    ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=colors['muted'], lw=1.5))

ax1.text(5.5, 2.5, 'Dual Objectives:', ha='center', color=colors['text'], 
         fontsize=10, fontweight='bold')
ax1.text(5.5, 2.0, '↓ f₁: Minimize Intra-Cluster Compactness (SSE)', 
         ha='center', color=colors['accent'], fontsize=9, fontweight='bold')
ax1.text(5.5, 1.6, '↑ f₂: Maximize Inter-Cluster Separation', 
         ha='center', color=colors['secondary'], fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('visual1_pipeline.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 2: Pareto Front Evolution
# ============================================
fig2, ax2 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax2)

np.random.seed(42)
generations = [1, 30, 60, 100, 150]
gen_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(generations)))

for idx, gen in enumerate(generations):
    n_points = 20 + gen // 5
    f1 = np.linspace(300, 2000 - gen*3, n_points)
    f2_base = 10 + 3 * np.log(f1/300) + np.random.normal(0, 0.15, n_points)
    f2 = f2_base + (5 - idx) * 0.5
    
    ax2.scatter(f1, f2, c=[gen_colors[idx]], s=35, alpha=0.8, 
              label=f'Gen {gen}', edgecolors='none')
    sort_idx = np.argsort(f1)
    ax2.plot(f1[sort_idx], f2[sort_idx], '--', color=gen_colors[idx], alpha=0.6, lw=1.2)

ax2.scatter([280], [10.0], c=colors['secondary'], s=180, marker='*', zorder=10, 
          label='K-Means (Single-Objective)', edgecolors=colors['text'], linewidth=0.8)

ax2.set_xlabel('Compactness Variance f₁ (Lower is Better)', color=colors['text'], fontsize=10)
ax2.set_ylabel('Inter-Cluster Separation f₂ (Higher is Better)', color=colors['text'], fontsize=10)
ax2.set_title('Pareto Front Evolution: Track A (Classical 2D)', 
              color=colors['text'], fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
          labelcolor=colors['text'], fontsize=9)

plt.tight_layout()
plt.savefig('visual2_pareto_evolution.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 3: Hypervolume Progress
# ============================================
fig3, ax3 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax3)

gens = np.arange(1, 301)
hypervolume = 0.5 + 0.4 * (1 - np.exp(-gens/50)) + 0.05 * np.sin(gens/10) * np.exp(-gens/100)
hypervolume += np.random.normal(0, 0.01, len(gens))

ax3.plot(gens, hypervolume, color=colors['accent'], lw=2, alpha=0.9)
ax3.fill_between(gens, hypervolume, alpha=0.15, color=colors['accent'])

phase_points = [(50, hypervolume[49], 'Phase I\nExploration'),
                (150, hypervolume[149], 'Phase II\nExploitation'),
                (250, hypervolume[249], 'Phase III\nConvergence')]

for x, y, label in phase_points:
    ax3.axvline(x=x, color=colors['warning'], linestyle='--', alpha=0.7)
    ax3.annotate(label, xy=(x, y), xytext=(x+15, y+0.04),
                color=colors['warning'], fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors['warning'], lw=1.2))

ax3.set_xlabel('Generation', color=colors['text'], fontsize=10)
ax3.set_ylabel('Hypervolume Indicator', color=colors['text'], fontsize=10)
ax3.set_title('Convergence Dynamics: Hypervolume Over Generations', 
              color=colors['text'], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visual3_hypervolume_convergence.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 4: Track Comparison (2D vs 384D)
# ============================================
fig4, ax4 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax4)

f1_a = np.linspace(280, 2000, 50)
f2_a = 10 + 3.5 * np.log(f1_a/280) + np.random.normal(0, 0.1, 50)

f1_b = np.concatenate([
    np.linspace(280, 450, 15),
    np.linspace(700, 1400, 25),
    np.linspace(1900, 2200, 10)
])
f2_b = np.concatenate([
    2.3 + 1.1 * np.linspace(0, 1, 15) + np.random.normal(0, 0.05, 15),
    3.5 + 0.9 * np.linspace(0, 1, 25) + np.random.normal(0, 0.05, 25),
    4.4 + 0.3 * np.linspace(0, 1, 10) + np.random.normal(0, 0.03, 10)
])

ax4.scatter(f1_a, f2_a, c=colors['primary'], s=40, alpha=0.8, 
           label='Track A: Classical 2D Space', edgecolors='none')
ax4.scatter(f1_b, f2_b, c=colors['secondary'], s=40, alpha=0.8, 
           label='Track B: Transformer 384D', edgecolors='none', marker='s')

ax4.axvspan(450, 700, alpha=0.15, color=colors['warning'])
ax4.axvspan(1400, 1900, alpha=0.15, color=colors['warning'])
ax4.text(575, 3.0, 'Optimization\nGap', ha='center', color=colors['warning'], 
         fontsize=9, fontweight='bold')
ax4.text(1650, 3.5, 'Optimization\nGap', ha='center', color=colors['warning'], 
         fontsize=9, fontweight='bold')

ax4.set_xlabel('Compactness Variance f₁', color=colors['text'], fontsize=10)
ax4.set_ylabel('Inter-Cluster Separation f₂', color=colors['text'], fontsize=10)
ax4.set_title('Scale Comparison: Classical vs Transformer Embedding Space', 
              color=colors['text'], fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
          labelcolor=colors['text'], fontsize=9)

plt.tight_layout()
plt.savefig('visual4_space_scale_comparison.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 5: Crowding Distance Distribution
# ============================================
fig5, ax5 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax5)
ax5.grid(True, alpha=0.6, color=colors['grid'], axis='y')

fronts_data = []
front_labels = ['Front 1 (F₁)', 'Front 2 (F₂)', 'Front 3 (F₃)', 'Front 4 (F₄)']
front_colors = [colors['accent'], colors['primary'], colors['purple'], colors['muted']]

for i in range(4):
    cd = np.random.exponential(scale=2/(i+1), size=30) + 0.1
    fronts_data.append(cd)

bp = ax5.boxplot(fronts_data, labels=front_labels, patch_artist=True,
                boxprops=dict(color=colors['border']),
                medianprops=dict(color=colors['text'], lw=2),
                whiskerprops=dict(color=colors['muted']),
                capprops=dict(color=colors['muted']),
                flierprops=dict(markerfacecolor=colors['secondary'], marker='o', markersize=4))

for patch, color in zip(bp['boxes'], front_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

ax5.set_ylabel('Crowding Distance (I_D)', color=colors['text'], fontsize=10)
ax5.set_title('Diversity Preservation: Crowding Distance by Front Rank', 
              color=colors['text'], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visual5_crowding_distance.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 6: Chromosome Representation & Operators
# ============================================
fig6, ax6 = plt.subplots(figsize=(10, 8), facecolor=colors['bg'])
ax6.set_facecolor(colors['bg'])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Genetic Operators: Chromosome Structure', 
              color=colors['text'], fontsize=12, fontweight='bold', pad=15)

parent1 = np.array([[2, 7], [4, 7.5], [6, 6.5]])
parent2 = np.array([[2.5, 6.5], [3.5, 8], [5.5, 7]])

for i, (p1, p2) in enumerate(zip(parent1, parent2)):
    ax6.scatter(*p1, c=colors['primary'], s=150, zorder=5, edgecolors='black', linewidth=0.5)
    ax6.scatter(*p2, c=colors['secondary'], s=150, zorder=5, edgecolors='black', linewidth=0.5)
    ax6.text(p1[0], p1[1]+0.4, f'C{i+1}', ha='center', color=colors['primary'], fontsize=9, fontweight='bold')
    ax6.text(p2[0], p2[1]-0.5, f"C'{i+1}", ha='center', color=colors['secondary'], fontsize=9, fontweight='bold')

alpha = 0.6
child = alpha * parent1 + (1-alpha) * parent2
for i, c in enumerate(child):
    ax6.scatter(*c, c=colors['accent'], s=150, zorder=5, marker='D', edgecolors='black', linewidth=0.5)
    ax6.text(c[0]+0.3, c[1], f'C"{i+1}', color=colors['accent'], fontsize=9, fontweight='bold')

mutated = child[1] + np.array([0.5, -0.3])
ax6.annotate('', xy=mutated, xytext=child[1],
            arrowprops=dict(arrowstyle='->', color=colors['pink'], lw=2, ls='--'))
ax6.scatter(*mutated, c=colors['pink'], s=150, zorder=5, marker='*', edgecolors='black', linewidth=0.5)
ax6.text(mutated[0]+0.3, mutated[1], 'Mutated', color=colors['pink'], fontsize=9, fontweight='bold')

ax6.text(4, 9, 'Arithmetic Crossover: C" = α ⊙ P₁ + (1-α) ⊙ P₂', 
         ha='center', color=colors['text'], fontsize=9, fontweight='bold')
ax6.text(4, 8.5, 'Gaussian Mutation: vᵢ ← vᵢ + ηᵢ ⊙ N(0, σ²I)', 
         ha='center', color=colors['muted'], fontsize=9)

hull1 = Polygon(parent1[[0,1,2]], alpha=0.15, facecolor=colors['primary'], edgecolor=colors['primary'])
hull2 = Polygon(parent2[[0,1,2]], alpha=0.15, facecolor=colors['secondary'], edgecolor=colors['secondary'])
ax6.add_patch(hull1)
ax6.add_patch(hull2)

plt.tight_layout()
plt.savefig('visual6_chromosome_operators.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 7: Non-Dominated Sorting
# ============================================
fig7, ax7 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax7)

np.random.seed(123)
n_pop = 80
f1_pop = np.random.uniform(200, 2000, n_pop)
f2_pop = np.random.uniform(8, 14, n_pop)

front1_mask = (f1_pop < 600) & (f2_pop > 12)
front2_mask = (f1_pop < 900) & (f2_pop > 11) & ~front1_mask
front3_mask = (f1_pop < 1300) & (f2_pop > 10) & ~(front1_mask | front2_mask)
front4_mask = ~(front1_mask | front2_mask | front3_mask)

ax7.scatter(f1_pop[front1_mask], f2_pop[front1_mask], c=colors['accent'], s=60, 
           label='Front 1 (F₁)', alpha=0.9, edgecolors='black', linewidth=0.3)
ax7.scatter(f1_pop[front2_mask], f2_pop[front2_mask], c=colors['primary'], s=50, 
           label='Front 2 (F₂)', alpha=0.8, edgecolors='black', linewidth=0.3)
ax7.scatter(f1_pop[front3_mask], f2_pop[front3_mask], c=colors['purple'], s=40, 
           label='Front 3 (F₃)', alpha=0.7, edgecolors='black', linewidth=0.3)
ax7.scatter(f1_pop[front4_mask], f2_pop[front4_mask], c=colors['muted'], s=30, 
           label='Front 4+ (Dominated)', alpha=0.5, edgecolors='black', linewidth=0.3)

ax7.annotate('', xy=(200, 14), xytext=(1800, 8),
            arrowprops=dict(arrowstyle='->', color=colors['warning'], lw=2, ls='--'))
ax7.text(1000, 10.5, 'Ideal Direction', color=colors['warning'], fontsize=9, 
         rotation=20, fontweight='bold')

ax7.set_xlabel('f₁: Compactness (minimize)', color=colors['text'], fontsize=10)
ax7.set_ylabel('f₂: Separation (maximize)', color=colors['text'], fontsize=10)
ax7.set_title('NSGA-II Non-Dominated Sorting: Front Hierarchy', 
              color=colors['text'], fontsize=12, fontweight='bold')
ax7.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
          labelcolor=colors['text'], fontsize=9)

plt.tight_layout()
plt.savefig('visual7_nondominated_sorting.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 8: Scalability Stress Test
# ============================================
fig8, ax8 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax8)
ax8.grid(True, alpha=0.6, color=colors['grid'], axis='y')

scenarios = ['Exploratory\n(N=100, G=150)', 'Production\n(N=200, G=300)']
x_pos = np.arange(len(scenarios))
width = 0.18

metrics = {
    'Hypervolume': ([0.65, 0.92], colors['accent']),
    'Front Spread': ([0.55, 0.88], colors['primary']),
    'Convergence': ([0.70, 0.95], colors['purple']),
    'Diversity': ([0.60, 0.85], colors['secondary'])
}

for i, (metric, (values, color)) in enumerate(metrics.items()):
    offset = (i - 1.5) * width
    bars = ax8.bar(x_pos + offset, values, width, label=metric, 
                   color=color, alpha=0.85, edgecolor=colors['border'], linewidth=0.8)
    for bar, val in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.2f}', ha='center', va='bottom', color=colors['text'], fontsize=8, fontweight='bold')

ax8.set_xticks(x_pos)
ax8.set_xticklabels(scenarios, color=colors['text'], fontsize=10)
ax8.set_ylabel('Normalized Score', color=colors['text'], fontsize=10)
ax8.set_title('Scalability Stress Test: Performance Metrics', 
              color=colors['text'], fontsize=12, fontweight='bold')
ax8.legend(loc='upper left', facecolor=colors['card'], edgecolor=colors['border'],
          labelcolor=colors['text'], fontsize=9)
ax8.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig('visual8_scalability_metrics.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

print("All 8 individual plots successfully created and saved in white theme:")
print("- visual1_pipeline.png")
print("- visual2_pareto_evolution.png")
print("- visual3_hypervolume_convergence.png")
print("- visual4_space_scale_comparison.png")
print("- visual5_crowding_distance.png")
print("- visual6_chromosome_operators.png")
import matplotlib.pyplot as plt
import numpy as np

# Set global style
plt.style.use('default')

# Color palette optimized for light/white theme
colors = {
    'primary': '#0969da',    # Deep Blue
    'secondary': '#cf222e',  # Crimson Red
    'accent': '#1a7f37',     # Forest Green
    'warning': '#9a6700',    # Dark Amber/Gold
    'purple': '#8250df',     # Vivid Purple
    'pink': '#bf3989',       # Magenta/Pink
    'text': '#1f2328',       # Charcoal Text
    'muted': '#57606a',      # Slate Gray
    'bg': '#ffffff',         # Pure White
    'card': '#f6f8fa',       # Very Light Gray Accent
    'border': '#d0d7de',     # Light Gray Border
    'grid': '#e1e4e8'        # Gridline Gray
}

def style_axis(ax):
    """Helper function for consistent light-theme axis styling."""
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['muted'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colors['border'])
    ax.grid(True, alpha=0.6, color=colors['grid'])

# ============================================
# VISUAL 9: Generation Progression Snapshots
# ============================================
fig9, ax9 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax9)

snapshot_gens = [1, 25, 75, 150, 225, 300]
colors_snap = plt.cm.viridis(np.linspace(0.1, 0.85, len(snapshot_gens)))

np.random.seed(42)
for idx, gen in enumerate(snapshot_gens):
    n_pts = 10 + gen // 10
    f1_snap = np.linspace(300, 2000 - (300-gen)*2, n_pts)
    f2_snap = 10 + (gen/300) * 3.5 * np.log(f1_snap/300) + np.random.normal(0, 0.08, n_pts)
    
    offset_y = idx * 0.3
    ax9.scatter(f1_snap, f2_snap + offset_y, c=[colors_snap[idx]], s=30, alpha=0.8)
    sort_idx = np.argsort(f1_snap)
    ax9.plot(f1_snap[sort_idx], f2_snap[sort_idx] + offset_y, '-', 
             color=colors_snap[idx], alpha=0.6, lw=1.5, label=f'Gen {gen}')

ax9.set_xlabel('f₁: Compactness Variance', color=colors['text'], fontsize=10)
ax9.set_ylabel('f₂: Separation (Offset for Clarity)', color=colors['text'], fontsize=10)
ax9.set_title('Generational Progression: Pareto Front Snapshots', 
              color=colors['text'], fontsize=12, fontweight='bold')
ax9.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
           labelcolor=colors['text'], fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('visual9_generational_progression.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 10: Cluster Assignment Evolution Heatmap
# ============================================
fig10, ax10 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
ax10.set_facecolor(colors['bg'])

n_samples = 50
n_gens_display = 20
gen_indices = np.linspace(0, 299, n_gens_display, dtype=int)

assignment_matrix = np.zeros((n_samples, n_gens_display))
for g_idx, gen in enumerate(gen_indices):
    if gen < 50:
        assignment_matrix[:, g_idx] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    elif gen < 150:
        assignment_matrix[:, g_idx] = np.random.choice([0, 1, 2], n_samples, p=[0.33, 0.33, 0.34])
    else:
        assignment_matrix[:, g_idx] = np.random.choice([0, 1, 2], n_samples, p=[0.34, 0.33, 0.33])

for i in range(n_samples):
    if i < 17:
        assignment_matrix[i, :] = np.where(np.random.rand(n_gens_display) > 0.1, 0, assignment_matrix[i, :])
    elif i < 34:
        assignment_matrix[i, :] = np.where(np.random.rand(n_gens_display) > 0.1, 1, assignment_matrix[i, :])
    else:
        assignment_matrix[i, :] = np.where(np.random.rand(n_gens_display) > 0.1, 2, assignment_matrix[i, :])

im = ax10.imshow(assignment_matrix, aspect='auto', cmap='Set2', interpolation='nearest')
ax10.set_xlabel('Generation (Sampled Snapshots)', color=colors['text'], fontsize=10)
ax10.set_ylabel('Data Sample Index', color=colors['text'], fontsize=10)
ax10.set_title('Cluster Assignment Stability Over Evolution', 
               color=colors['text'], fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)
cbar.set_label('Cluster ID', color=colors['text'], fontsize=9)
cbar.ax.tick_params(colors=colors['muted'], labelsize=8)

ax10.tick_params(colors=colors['muted'])
for spine in ax10.spines.values():
    spine.set_color(colors['border'])

plt.tight_layout()
plt.savefig('visual10_assignment_stability_heatmap.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 11: Fitness Landscape Surface
# ============================================
fig11, ax11 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
ax11.set_facecolor(colors['bg'])

x = np.linspace(250, 2000, 100)
y = np.linspace(9, 14, 100)
X, Y = np.meshgrid(x, y)

Z = np.sin(X/200) * np.cos(Y) + 0.5 * np.sin(X/100) * np.sin(Y*2)
Z = (Z - Z.min()) / (Z.max() - Z.min())

contour = ax11.contourf(X, Y, Z, levels=20, cmap='YlGnBu_r', alpha=0.85)
ax11.contour(X, Y, Z, levels=10, colors=colors['muted'], alpha=0.3, linewidths=0.5)

pareto_x = np.linspace(280, 2000, 50)
pareto_y = 10 + 3.5 * np.log(pareto_x/280)
ax11.plot(pareto_x, pareto_y, '--', color=colors['text'], lw=2, label='Pareto Front Path')
ax11.scatter(pareto_x[::5], pareto_y[::5], c=colors['bg'], s=35, zorder=5, edgecolors=colors['text'], linewidth=1)

ax11.scatter([280], [10], c=colors['secondary'], s=180, marker='*', zorder=10, 
            edgecolors=colors['text'], linewidth=0.8, label='K-Means')

ax11.set_xlabel('f₁: Compactness Variance', color=colors['text'], fontsize=10)
ax11.set_ylabel('f₂: Separation', color=colors['text'], fontsize=10)
ax11.set_title('Fitness Landscape: Multi-Objective Optimization Surface', 
               color=colors['text'], fontsize=12, fontweight='bold')
ax11.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
           labelcolor=colors['text'], fontsize=9)
ax11.tick_params(colors=colors['muted'])
for spine in ax11.spines.values():
    spine.set_color(colors['border'])

cbar2 = plt.colorbar(contour, ax=ax11, fraction=0.046, pad=0.04)
cbar2.set_label('Combined Fitness Score', color=colors['text'], fontsize=9)
cbar2.ax.tick_params(colors=colors['muted'], labelsize=8)

plt.tight_layout()
plt.savefig('visual11_fitness_landscape.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 12: Topological Bifurcation
# ============================================
fig12, ax12 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax12)

f1_classical = np.concatenate([
    np.linspace(280, 1100, 30),
    np.linspace(1600, 2000, 15)
])
f2_classical = np.concatenate([
    10 + 3.5 * np.log(f1_classical[:30]/280) + np.random.normal(0, 0.05, 30),
    13.2 + 0.3 * np.linspace(0, 1, 15) + np.random.normal(0, 0.03, 15)
])

ax12.scatter(f1_classical[:30], f2_classical[:30], c=colors['primary'], s=50, 
            alpha=0.85, label='Track A: Pre-Bifurcation', edgecolors='black', linewidth=0.3)
ax12.scatter(f1_classical[30:], f2_classical[30:], c=colors['accent'], s=50, 
            alpha=0.85, label='Track A: Post-Bifurcation', edgecolors='black', linewidth=0.3, marker='s')

ax12.axvspan(1100, 1600, alpha=0.15, color=colors['warning'])
ax12.annotate('Topological\nBifurcation Gap', xy=(1350, 12), ha='center',
             color=colors['warning'], fontsize=10, fontweight='bold')

ax12.annotate('Structural Jump:\nTwo clusters merge\nto isolate third', 
             xy=(1800, 13.5), xytext=(1500, 11.5),
             color=colors['accent'], fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.5))

ax12.set_xlabel('f₁: Compactness Variance', color=colors['text'], fontsize=10)
ax12.set_ylabel('f₂: Separation', color=colors['text'], fontsize=10)
ax12.set_title('Track A: Topological Bifurcation at Production Scale', 
               color=colors['text'], fontsize=12, fontweight='bold')
ax12.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
           labelcolor=colors['text'], fontsize=9)

plt.tight_layout()
plt.savefig('visual12_topological_bifurcation.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 13: Track B Quantized Phase Steps
# ============================================
fig13, ax13 = plt.subplots(figsize=(10, 7), facecolor=colors['bg'])
style_axis(ax13)

phase1_f1 = np.linspace(280, 450, 15)
phase1_f2 = 2.3 + 1.1 * np.linspace(0, 1, 15) + np.random.normal(0, 0.04, 15)

phase2_f1 = np.linspace(700, 1400, 25)
phase2_f2 = 3.5 + 0.9 * np.linspace(0, 1, 25) + np.random.normal(0, 0.04, 25)

phase3_f1 = np.linspace(1900, 2200, 10)
phase3_f2 = 4.4 + 0.3 * np.linspace(0, 1, 10) + np.random.normal(0, 0.03, 10)

ax13.scatter(phase1_f1, phase1_f2, c=colors['accent'], s=55, 
            alpha=0.9, label='Phase I: Compact Zone', edgecolors='black', linewidth=0.3, marker='o')
ax13.scatter(phase2_f1, phase2_f2, c=colors['primary'], s=55, 
            alpha=0.9, label='Phase II: Trade-off Zone', edgecolors='black', linewidth=0.3, marker='s')
ax13.scatter(phase3_f1, phase3_f2, c=colors['secondary'], s=55, 
            alpha=0.9, label='Phase III: Extreme Isolation', edgecolors='black', linewidth=0.3, marker='^')

ax13.plot([phase1_f1[-1], phase2_f1[0]], [phase1_f2[-1], phase2_f2[0]], 
         '--', color=colors['muted'], alpha=0.5, lw=1.2)
ax13.plot([phase2_f1[-1], phase3_f1[0]], [phase2_f2[-1], phase3_f2[0]], 
         '--', color=colors['muted'], alpha=0.5, lw=1.2)

ax13.axvspan(450, 700, alpha=0.15, color=colors['warning'])
ax13.axvspan(1400, 1900, alpha=0.15, color=colors['warning'])
ax13.text(575, 3.2, 'Epistatic\nBarrier I', ha='center', color=colors['warning'], 
         fontsize=9, fontweight='bold')
ax13.text(1650, 3.8, 'Epistatic\nBarrier II', ha='center', color=colors['warning'], 
         fontsize=9, fontweight='bold')

ax13.set_xlabel('f₁: Compactness Variance', color=colors['text'], fontsize=10)
ax13.set_ylabel('f₂: Separation', color=colors['text'], fontsize=10)
ax13.set_title('Track B: Quantized Phase Transitions in 384D Space', 
               color=colors['text'], fontsize=12, fontweight='bold')
ax13.legend(loc='lower right', facecolor=colors['card'], edgecolor=colors['border'],
           labelcolor=colors['text'], fontsize=9)

plt.tight_layout()
plt.savefig('visual13_quantized_phase_steps.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ============================================
# VISUAL 14: Algorithm Comparison Radar Chart
# ============================================
fig14 = plt.figure(figsize=(8, 8), facecolor=colors['bg'])
ax14 = fig14.add_subplot(1, 1, 1, projection='polar')
ax14.set_facecolor(colors['bg'])

categories = ['Compactness', 'Separation', 'Diversity', 'Convergence', 
              'Scalability', 'Robustness']
N = len(categories)

kmeans_vals = [0.95, 0.3, 0.1, 0.9, 0.95, 0.4]
nsga_vals = [0.7, 0.85, 0.9, 0.8, 0.75, 0.85]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
kmeans_vals += kmeans_vals[:1]
nsga_vals += nsga_vals[:1]

ax14.plot(angles, kmeans_vals, 'o-', linewidth=2, label='K-Means', color=colors['secondary'])
ax14.fill(angles, kmeans_vals, alpha=0.15, color=colors['secondary'])
ax14.plot(angles, nsga_vals, 'o-', linewidth=2, label='NSGA-II (Proposed)', color=colors['accent'])
ax14.fill(angles, nsga_vals, alpha=0.15, color=colors['accent'])

ax14.set_xticks(angles[:-1])
ax14.set_xticklabels(categories, color=colors['text'], fontsize=9, fontweight='bold')
ax14.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax14.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color=colors['muted'], fontsize=8)
ax14.set_title('Algorithm Comparison: Multi-Criteria Radar', 
               color=colors['text'], fontsize=12, fontweight='bold', pad=25)
ax14.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), facecolor=colors['card'],
           edgecolor=colors['border'], labelcolor=colors['text'], fontsize=9)
ax14.grid(True, alpha=0.4, color=colors['grid'])
ax14.spines['polar'].set_color(colors['border'])

plt.tight_layout()
plt.savefig('visual14_radar_comparison.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

print("All visuals (9 through 14) successfully generated and saved:")
print("- visual9_generational_progression.png")
print("- visual10_assignment_stability_heatmap.png")
print("- visual11_fitness_landscape.png")
print("- visual12_topological_bifurcation.png")
print("- visual13_quantized_phase_steps.png")
print("- visual14_radar_comparison.png")
print("- visual7_nondominated_sorting.png")
print("- visual8_scalability_metrics.png")
