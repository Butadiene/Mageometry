"""Draw flux-tube cross sections for the transverse-gradient decomposition.

Generate editable SVG, PDF, and PNG figures with ``.[viz]`` installed::

    python benchmark/transverse_decomposition_figure.py

The curves follow exp(s M) in a transported normal plane. A gently curved,
planar centreline supplies a defined Frenet frame with zero torsion. These
are local, thin-tube illustrations, not a finite-volume MHD simulation.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
MODES = ('theta', 'delta_g', 'beta_g', 'alpha')
COLORS = ('#008577', '#2875b9', '#2875b9', '#8552a3')
INK = '#203448'
MUTED = '#697888'
MARKER = '#df861e'
LENGTH = 2.7
CURVATURE = 0.12
RADIUS = 0.48
STRENGTH = 0.48
SCREEN_X = np.array([np.sqrt(3.) / 2., -.5, 0.])
SCREEN_Y = np.array([.25, np.sqrt(3.) / 4., np.sqrt(3.) / 2.])
SCREEN_DEPTH = np.cross(SCREEN_X, SCREEN_Y)


def mode_matrix(mode):
    """Return the dimensionless generator ell*M for one isolated mode."""
    generators = {
        'theta': [[1., 0.], [0., 1.]],
        'delta_g': [[1., 0.], [0., -1.]],
        'beta_g': [[0., 1.], [1., 0.]],
        'alpha': [[0., -1.], [1., 0.]],
    }
    strength = np.pi / 3 if mode == 'alpha' else STRENGTH
    return strength * np.array(generators[mode])


def transverse_map(mode, fraction):
    """Return the exact constant-generator map at a fraction of the segment."""
    return expm(fraction * mode_matrix(mode))


def centre_frame(s):
    """Return a planar arc and its right-handed, torsion-free Frenet frame."""
    s = np.asarray(s, dtype=float)
    angle = CURVATURE * s
    zero = np.zeros_like(s)
    centre = np.stack(((1. - np.cos(angle)) / CURVATURE,
                       zero, np.sin(angle) / CURVATURE), axis=-1)
    tangent = np.stack((np.sin(angle), zero, np.cos(angle)), axis=-1)
    normal = np.stack((np.cos(angle), zero, -np.sin(angle)), axis=-1)
    binormal = np.broadcast_to([0., 1., 0.], centre.shape)
    return centre, tangent, normal, binormal


def tube_points(mode, fraction, angles):
    """Embed the mapped initial circle in each perpendicular normal plane."""
    centre, _, normal, binormal = centre_frame(LENGTH * fraction)
    initial = RADIUS * np.array([np.cos(angles), np.sin(angles)])
    offsets = transverse_map(mode, fraction) @ initial
    return centre + offsets[0, ..., None] * normal + offsets[1, ..., None] * binormal


def project(points):
    return np.stack((points @ SCREEN_X, points @ SCREEN_Y), axis=-1)


def check_geometry():
    """Check areas, rotation, strain axes, handedness, and local gradients."""
    fractions = np.linspace(0., 1., 7)
    centre, tangent, normal, binormal = centre_frame(LENGTH * fractions)
    np.testing.assert_allclose(np.cross(tangent, normal), binormal, atol=1e-14)
    np.testing.assert_allclose(np.sum(tangent * normal, axis=-1), 0., atol=1e-14)
    for mode in MODES:
        generator = mode_matrix(mode)
        for fraction in fractions:
            mapping = transverse_map(mode, fraction)
            np.testing.assert_allclose(np.linalg.det(mapping),
                                       np.exp(fraction * np.trace(generator)), rtol=1e-13)
            np.testing.assert_allclose(mapping @ transverse_map(mode, -fraction),
                                       np.eye(2), atol=1e-13)
        if mode == 'alpha':
            np.testing.assert_allclose(mapping.T @ mapping, np.eye(2), atol=1e-13)
            np.testing.assert_allclose(np.arctan2(mapping[1, 0], mapping[0, 0]), generator[1, 0])
        if mode in ('delta_g', 'beta_g'):
            np.testing.assert_allclose(mapping.T, mapping, atol=1e-13)
            direction = np.array([1., 0.]) if mode == 'delta_g' else np.array([1., 1.])
            np.testing.assert_allclose(mapping @ direction, np.exp(STRENGTH) * direction)
        # At the centreline, projecting dT/d(xi) recovers the requested M.
        s = LENGTH * .4
        _, t, n, b = centre_frame(s)
        frame = np.column_stack((n, b))
        matrix = generator / LENGTH

        def unit_direction(xi):
            vector = (1. - CURVATURE * xi[0]) * t + frame @ matrix @ xi
            return vector / np.linalg.norm(vector)

        step = 1e-6
        gradient = np.column_stack([
            (unit_direction(step * e) - unit_direction(-step * e)) / (2 * step)
            for e in np.eye(2)])
        np.testing.assert_allclose(frame.T @ gradient, matrix, atol=1e-9)
    print('Geometry checks passed: area, strain, rotation, frame, and transverse gradients.')


def arrow(ax, start, end, color=INK, width=1.3, scale=10, **kwargs):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', color=color,
                                linewidth=width, mutation_scale=scale, **kwargs))


def draw_tube(ax, mode, color, show_frame=False):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    angles = np.linspace(0., 2 * np.pi, 49)
    fractions = np.linspace(0., 1., 30)
    rings = np.array([tube_points(mode, f, angles) for f in fractions])
    polygons, depths, shades = [], [], []
    rgb = np.array(to_rgb(color))
    for j in range(len(fractions) - 1):
        for i in range(len(angles) - 1):
            vertices = rings[[j, j + 1, j + 1, j], [i, i, i + 1, i + 1]]
            polygons.append(project(vertices))
            depths.append(np.mean(vertices @ SCREEN_DEPTH))
            tint = .76 + .12 * np.cos(angles[i] + .4)
            shades.append(tuple(tint + (1 - tint) * rgb))
    order = np.argsort(depths)
    ax.add_collection(PolyCollection([polygons[i] for i in order],
                                     facecolors=[shades[i] for i in order],
                                     edgecolors='none', alpha=.68, zorder=1))
    for angle in np.linspace(0., 2 * np.pi, 12, endpoint=False):
        path = project(np.array([tube_points(mode, f, angle) for f in fractions]))
        ax.plot(*path.T, color=color, linewidth=.8, alpha=.55, zorder=2)
    for fraction in (0., .5, 1.):
        ring = project(tube_points(mode, fraction, angles))
        ax.plot(*ring.T, color=color, linewidth=1.6 if fraction in (0., 1.) else .65,
                alpha=.9 if fraction in (0., 1.) else .5, zorder=3)

    # A field-line marker makes rotation visible even when the tube stays circular.
    marker = project(np.array([tube_points(mode, f, 0.) for f in fractions]))
    ax.plot(*marker.T, color=MARKER, linewidth=2.7, zorder=5)
    ax.scatter(*marker[[0, -1]].T, s=24, color=MARKER,
               edgecolors='white', linewidths=.8, zorder=6)
    centre, t, n, b = centre_frame(LENGTH * fractions)
    axis = project(centre)
    ax.plot(*axis.T, color=MUTED, linewidth=.8, linestyle=(0, (3, 3)), alpha=.7, zorder=4)
    arrow(ax, axis[-5], project(centre[-1] + .34 * t[-1]), MUTED, scale=9, zorder=5)
    ax.text(*(project(centre[-1] + .43 * t[-1]) + [.06, 0]), r'$s$',
            fontsize=12, color=MUTED, va='center')
    ax.text(-.7, -.25, r'$s_0$', fontsize=11, color=MUTED)
    end = project(centre[-1])
    ax.text(end[0] + .86, end[1], r'$s_0+\ell$', fontsize=11, color=MUTED, va='center')
    if show_frame:
        origin = project(centre[0])
        for direction, label in ((n[0], r'$\mathbf{n}$'), (b[0], r'$\mathbf{b}$'), (t[0], r'$\mathbf{T}$')):
            endpoint = project(centre[0] + .64 * direction)
            arrow(ax, origin, endpoint, MUTED, width=1., scale=8, zorder=7)
            ax.text(*(origin + 1.19 * (endpoint - origin)), label, fontsize=10,
                    color=INK, ha='center', va='center', zorder=8)
    ax.set(xlim=(-1.05, 1.6), ylim=(-.5, 3.05), aspect='equal')
    ax.axis('off')


def draw_section(ax, mode, color):
    from matplotlib.patches import Polygon

    angles = np.linspace(0., 2 * np.pi, 361)
    circle = np.array([np.cos(angles), np.sin(angles)])
    mapping = transverse_map(mode, 1.)
    final = mapping @ circle
    ax.add_patch(Polygon(final.T, closed=True, facecolor=color, alpha=.055, edgecolor='none'))
    for coord in (-.5, 0., .5):
        span = np.linspace(-np.sqrt(1 - coord**2), np.sqrt(1 - coord**2), 50)
        for line in (np.array([span, np.full_like(span, coord)]),
                     np.array([np.full_like(span, coord), span])):
            transformed = mapping @ line
            ax.plot(*transformed, color=color, linewidth=.65, alpha=.34, zorder=1)
    ax.plot(*final, color=color, linewidth=2.2, zorder=3)
    ax.plot(*circle, color='#9da8b2', linewidth=1.1, linestyle=(0, (3, 3)), zorder=4)
    for endpoint, label, shift in (([1.86, 0.], r'$\mathbf{n}$', [0., -.18]),
                                   ([0., 1.86], r'$\mathbf{b}$', [.14, 0.])):
        arrow(ax, [0., 0.], endpoint, '#8595a2', width=.8, scale=7, zorder=2)
        ax.text(*(np.array(endpoint) + shift), label, fontsize=11,
                color=MUTED, ha='center', va='center')

    # The initial marked radius, and its image, are physical labels, not a rotating basis.
    ax.plot([0., 1.], [0., 0.], color=MARKER, linewidth=1.2, linestyle=(0, (2, 3)), zorder=5)
    point = mapping @ [1., 0.]
    ax.plot([0., point[0]], [0., point[1]], color=MARKER, linewidth=1.5, zorder=6)
    ax.scatter([1.], [0.], s=24, facecolors='white', edgecolors=MARKER, zorder=7)
    ax.scatter(*point, s=36, color=MARKER, edgecolors='white', linewidths=.65, zorder=8)
    if mode == 'alpha':
        path = np.array([transverse_map(mode, f) @ [1., 0.] for f in np.linspace(0., 1., 40)])
        ax.plot(*path.T, color=MARKER, linewidth=2.3, zorder=8)
        arrow(ax, path[-6], path[-1], MARKER, width=1.5, scale=10, zorder=9)
    elif mode == 'theta':
        for angle in (np.pi / 2, np.pi, 3 * np.pi / 2):
            initial = np.array([np.cos(angle), np.sin(angle)])
            arrow(ax, initial, mapping @ initial, color, width=1.2, scale=10, zorder=8)
    else:
        angle = 0. if mode == 'delta_g' else np.pi / 4
        for phi in (angle, angle + np.pi / 2, angle + np.pi, angle + 3 * np.pi / 2):
            initial = np.array([np.cos(phi), np.sin(phi)])
            arrow(ax, initial, mapping @ initial, color, width=1.2, scale=10, zorder=8)
    ax.set(xlim=(-1.97, 2.02), ylim=(-1.87, 2.02), aspect='equal')
    ax.axis('off')


def matrix(fig, centre_x, centre_y, entries, color=INK, size=16, width=.06, height=.049):
    """Draw a small editable 2-by-2 matrix without an external TeX dependency."""
    from matplotlib.lines import Line2D

    for i in range(2):
        for j in range(2):
            fig.text(centre_x + (j - .5) * width * .56,
                     centre_y + (.5 - i) * height * .56,
                     entries[i][j], fontsize=size, color=color, ha='center', va='center')
    for sign in (-1, 1):
        x = centre_x + sign * width / 2
        fig.add_artist(Line2D([x - sign * .004, x, x, x - sign * .004],
                              [centre_y - height / 2, centre_y - height / 2,
                               centre_y + height / 2, centre_y + height / 2],
                              transform=fig.transFigure, color=color, linewidth=1.25))


def make_figure(language, cjk_font=None):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D

    ja = language == 'ja'
    japanese_font = cjk_font or Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if ja and not japanese_font.exists():
        raise RuntimeError('Japanese rendering needs a CJK font; use --cjk-font to select one.')
    font = None
    if ja:
        fontManager.addfont(str(japanese_font))
        family = FontProperties(fname=str(japanese_font)).get_name()
        font = FontProperties(family=['DejaVu Sans', family])
    fig = plt.figure(figsize=(18, 10.125), facecolor='white')

    def label(x, y, english, japanese=None, size=12, color=INK, ha='left', **kwargs):
        if ja and japanese:
            kwargs.pop('weight', None)
        return fig.text(x, y, japanese if ja and japanese else english,
                        fontsize=size, color=color, ha=ha,
                        fontproperties=font if ja and japanese else None, **kwargs)

    label(.04, .951, 'How a magnetic flux tube changes across neighbouring sections',
          '磁束管の横断面は、磁力線に沿ってどのように変わるか', size=25, weight='bold')
    label(.04, .916, 'Three tensor parts. Each column isolates one positive coefficient.',
          '各列は１つの係数だけを正にし、ほかの３つはゼロとした例', size=12.5, color=MUTED)
    fig.text(.96, .916, r'$\mathbf{T}=\mathbf{B}/B,\quad \mathbf{b}=\mathbf{T}\times\mathbf{n}$',
             fontsize=14, color=INK, ha='right')

    y = .858
    fig.text(.075, y, r'$M=$', fontsize=23, color=INK, va='center')
    matrix(fig, .167, y, [[r'$a$', r'$q$'], [r'$p$', r'$d$']], width=.065)
    fig.text(.223, y, r'$=$', fontsize=23, color=INK, va='center')
    fig.text(.288, y, r'$\frac{\theta}{2}$', fontsize=25, color=COLORS[0], va='center')
    matrix(fig, .358, y, [[r'$1$', r'$0$'], [r'$0$', r'$1$']], color=COLORS[0])
    fig.text(.42, y, r'$+$', fontsize=23, color=INK, va='center')
    fig.text(.477, y, r'$\frac{1}{2}$', fontsize=25, color=COLORS[1], va='center')
    matrix(fig, .558, y, [[r'$\delta_g$', r'$\beta_g$'], [r'$\beta_g$', r'$-\delta_g$']],
           color=COLORS[1], width=.092)
    fig.text(.64, y, r'$+$', fontsize=23, color=INK, va='center')
    fig.text(.701, y, r'$\frac{\alpha}{2}$', fontsize=25, color=COLORS[3], va='center')
    matrix(fig, .78, y, [[r'$0$', r'$-1$'], [r'$1$', r'$0$']], color=COLORS[3], width=.075)

    centres = [.14, .38, .62, .86]
    for left, right, color, english, japanese in (
            (.04, .24, COLORS[0], 'ISOTROPIC DILATION', '等方的な膨張・収縮'),
            (.275, .725, COLORS[1], 'SYMMETRIC STRAIN  /  BIAXIAL SPLAY', '対称・トレースゼロの変形  /  biaxial splay'),
            (.76, .96, COLORS[3], 'LOCAL ROTATION', '反対称の局所回転')):
        fig.add_artist(Line2D([left, right], [.79, .79], transform=fig.transFigure,
                              color=color, linewidth=2.5))
        label((left + right) / 2, .764, english, japanese, size=11.5, color=color, ha='center')
    titles = [('Uniform expansion', '全方向への膨張'),
              ('Stretch along n; squeeze along b', 'n 方向に伸長、b 方向に圧縮'),
              ('Stretch and squeeze at 45 degrees', '45° 方向の伸長・圧縮'),
              ('Rotate marked field lines', '目印を付けた磁力線が回転')]
    symbols = (r'$\theta>0$', r'$\delta_g>0$', r'$\beta_g>0$', r'$\alpha>0$')
    for i, (mode, color, x) in enumerate(zip(MODES, COLORS, centres)):
        label(x, .73, *titles[i], size=12.1, ha='center')
        fig.text(x, .699, symbols[i], fontsize=18, color=color, ha='center')
        draw_tube(fig.add_axes([x - .108, .455, .216, .235]), mode, color, show_frame=i == 0)
        draw_section(fig.add_axes([x - .098, .191, .196, .237]), mode, color)

    label(.04, .434, 'Normal-plane view', '横断面を正面から見る', size=10.5, color=MUTED)
    label(.96, .434, 'n right, b up; T points towards the viewer',
          'n：右、b：上、T：手前向き', size=10.5, color=MUTED, ha='right')

    definitions = (r'$\theta=a+d$', r'$\delta_g=a-d$', r'$\beta_g=p+q=\mathcal{D}/B$',
                   r'$\alpha=p-q=\mu_0j_\parallel/B$')
    effects = (r'$A_1/A_0=e^{\theta\ell}$', r'$A_1=A_0$', r'$A_1=A_0$',
               r'$\Delta\phi=\alpha\ell/2,\quad A_1=A_0$')
    for x, color, definition, effect in zip(centres, COLORS, definitions, effects):
        fig.text(x, .159, definition, fontsize=16, color=color, ha='center')
        fig.text(x, .127, effect, fontsize=13, color=INK, ha='center')

    label(.04, .101,
          r'$A_0, A_1$: areas normal to $\mathbf{T}$ at $s_0$ and $s_0+\ell$;    '
          r'$\ell$: arc length between sections along the reference line.',
          'A₀, A₁：s₀ と s₀ + ℓ での T に垂直な断面積。　'
          'ℓ：両断面間の基準磁力線に沿った弧長。',
          size=10.5, color=INK)
    label(.04, .075, 'Dashed grey: initial section     Colour: mapped section     Orange: the same labelled field line',
          '灰色の破線：初期断面　　実線：移動先の断面　　オレンジ：同じ磁力線を追う目印',
          size=11, color=MUTED)
    label(.96, .075, 'Negative signs reverse the illustrated map.',
          '符号が負の場合は逆向きの写像。', size=10.5, color=MUTED, ha='right')
    fig.add_artist(Line2D([.04, .96], [.050, .050], transform=fig.transFigure,
                          color='#d8e0e7', linewidth=.8))
    fig.text(.04, .018, r'$\boldsymbol{\xi}(s_0+\ell)=\exp(\ell M)\,\boldsymbol{\xi}(s_0)$',
             fontsize=14, color=INK)
    label(.365, .020, 'Constant local coefficients; transported frame (torsion = 0 here).  s is distance, not time.',
          '局所係数を固定した細い管の模式図。図の基底はねじれなし（τ = 0）。s は時間ではなく距離。',
          size=10.3, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/images')
    parser.add_argument('--language', choices=('en', 'ja', 'both'), default='en')
    parser.add_argument('--cjk-font', type=Path, help='Japanese font file for --language ja/both')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    check_geometry()
    if args.check_only:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language in ('en', 'ja') if args.language == 'both' else (args.language,):
        figure = make_figure(language, args.cjk_font)
        stem = 'transverse-decomposition' + ('-ja' if language == 'ja' else '')
        for extension in ('svg', 'pdf', 'png'):
            path = args.output_dir / f'{stem}.{extension}'
            figure.savefig(path, dpi=180, facecolor='white')
            if extension == 'svg':
                # Matplotlib leaves spaces at the ends of SVG path-data lines.
                lines = path.read_text(encoding='utf-8').splitlines()
                path.write_text('\n'.join(line.rstrip() for line in lines) + '\n',
                                encoding='utf-8')
            print(f'Saved {path}')
        plt.close(figure)


if __name__ == '__main__':
    main()
