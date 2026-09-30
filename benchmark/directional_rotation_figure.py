"""Draw local neighbour rotation and its uniform azimuthal mean.

Run with ``.[viz]`` installed::

    python benchmark/directional_rotation_figure.py --language both

This original vector illustration uses the transverse gradient and exp(s M),
with the same planar reference curve and visual conventions as the companion
transverse-decomposition figure. No external image is used.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from transverse_decomposition_figure import (
    ROOT, INK, MUTED, MARKER, LENGTH, CURVATURE, RADIUS,
    SCREEN_DEPTH, arrow, centre_frame, matrix, project,
)


MEAN = '#8552a3'
RATE = '#2875b9'
BINORMAL = '#008577'
L0 = 3.  # Reference length used to make the plotted rates dimensionless.
ALPHA, BETA_G, DELTA_G, THETA = 1., 1.5, .5, 0.  # Rates multiplied by L0.
PHI = np.pi / 5


def transverse_matrix(alpha, beta_g, delta_g, theta=0.):
    """Return M in the output-row (n, b) convention."""
    return .5 * np.array([[theta + delta_g, beta_g - alpha],
                          [beta_g + alpha, theta - delta_g]])


def omega(phi, alpha=ALPHA, beta_g=BETA_G, delta_g=DELTA_G):
    """Return the instantaneous angular rate for a unit separation direction."""
    return .5 * (alpha + beta_g * np.cos(2 * phi) - delta_g * np.sin(2 * phi))


def tube_points(s, angles):
    centre, _, normal, binormal = centre_frame(s)
    initial = RADIUS * np.array([np.cos(angles), np.sin(angles)])
    offset = expm(s / L0 * transverse_matrix(ALPHA, BETA_G, DELTA_G, THETA)) @ initial
    return centre + offset[0, ..., None] * normal + offset[1, ..., None] * binormal


def check_geometry():
    """Check signed rates against matrix action, 3D gradients and finite maps."""
    phi = np.linspace(0., 2 * np.pi, 4096, endpoint=False)
    directions = np.array([np.cos(phi), np.sin(phi)])
    rng = np.random.default_rng(19)
    examples = [(ALPHA, BETA_G, DELTA_G, THETA), (1., 0., 0., 0.),
                (0., 1., 0., 0.), (0., 0., 1., 0.), *rng.normal(size=(8, 4))]
    _, tangent, normal, binormal = centre_frame(.4 * LENGTH)
    frame = np.column_stack((normal, binormal))
    np.testing.assert_allclose(np.cross(tangent, normal), binormal, atol=1e-14)
    for alpha, beta_g, delta_g, theta in examples:
        matrix = transverse_matrix(alpha, beta_g, delta_g, theta)
        mapped = matrix @ directions
        direct = directions[0] * mapped[1] - directions[1] * mapped[0]
        expected = omega(phi, alpha, beta_g, delta_g)
        np.testing.assert_allclose(direct, expected, atol=2e-15)
        np.testing.assert_allclose(np.mean(direct), alpha / 2, atol=2e-15)
        np.testing.assert_allclose([direct[0], direct[len(phi) // 4]],
                                   [matrix[1, 0], -matrix[0, 1]], atol=1e-14)
        np.testing.assert_allclose(omega(phi + np.pi, alpha, beta_g, delta_g),
                                   expected, atol=4e-15)
        step = 1e-6
        angles = []
        for sign in (-1, 1):
            neighbour = expm(sign * step * matrix) @ directions
            angles.append(np.arctan2(
                directions[0] * neighbour[1] - directions[1] * neighbour[0],
                np.sum(directions * neighbour, axis=0)))
        np.testing.assert_allclose((angles[1] - angles[0]) / (2 * step),
                                   expected, atol=2e-10)

        def unit_tangent(xi):
            vector = (1. - CURVATURE * xi[0]) * tangent + frame @ matrix @ xi
            return vector / np.linalg.norm(vector)

        gradient = np.column_stack([
            (unit_tangent(step * e) - unit_tangent(-step * e)) / (2 * step)
            for e in np.eye(2)])
        physical_e = (frame @ directions).T
        directional_derivative = (gradient @ directions).T
        cross_rate = np.cross(physical_e, directional_derivative) @ tangent
        np.testing.assert_allclose(cross_rate, expected, atol=2e-10)
        np.testing.assert_allclose(np.linalg.det(expm(.7 * matrix)), np.exp(.7 * theta), rtol=1e-13)
    assert omega(0.) > 0 and omega(np.pi / 2) < 0 and ALPHA > 0
    np.testing.assert_allclose(omega(0.) + omega(np.pi / 2), ALPHA)
    np.testing.assert_allclose(omega(0.) - omega(np.pi / 2), BETA_G)
    print('Geometry checks passed: signs, matrix/3D definitions, local angles, mean, and finite maps.')


def draw_tube(ax):
    from matplotlib.collections import PolyCollection

    angles = np.linspace(0., 2 * np.pi, 65)
    distances = np.linspace(0., LENGTH, 45)
    rings = np.array([tube_points(s, angles) for s in distances])
    polygons, depths, shades = [], [], []
    for j in range(len(distances) - 1):
        for i in range(len(angles) - 1):
            vertices = rings[[j, j + 1, j + 1, j], [i, i, i + 1, i + 1]]
            polygons.append(project(vertices))
            depths.append(np.mean(vertices @ SCREEN_DEPTH))
            tint = .90 + .04 * np.cos(angles[i] + .4)
            shades.append((tint - .06, tint - .025, tint))
    order = np.argsort(depths)
    ax.add_collection(PolyCollection([polygons[i] for i in order],
                                     facecolors=[shades[i] for i in order],
                                     edgecolors='none', alpha=.62, zorder=1))
    for phi in np.linspace(0., 2 * np.pi, 12, endpoint=False):
        path = project(np.array([tube_points(s, phi) for s in distances]))
        ax.plot(*path.T, color='#8ba1b5', linewidth=.85, alpha=.5, zorder=2)
    for s in (0., LENGTH):
        ring = project(tube_points(s, angles))
        ax.plot(*ring.T, color='#6c879e', linewidth=1.5, zorder=3)
    for phi, color in ((0., MARKER), (np.pi / 2, BINORMAL)):
        path = project(np.array([tube_points(s, phi) for s in distances]))
        ax.plot(*path.T, color=color, linewidth=3., zorder=6)
        ax.scatter(*path[[0, -1]].T, color=color, edgecolors='white', linewidths=.8,
                   s=36, zorder=7)
    centre, tangent, normal, binormal = centre_frame(distances)
    path = project(centre)
    ax.plot(*path.T, color=INK, linewidth=2., zorder=5)
    arrow(ax, path[-5], project(centre[-1] + .28 * tangent[-1]), INK, scale=11, zorder=8)
    ax.text(*(project(centre[-1] + .34 * tangent[-1]) + [.08, 0.]),
            r'$s$', fontsize=15, color=INK)
    base = project(centre[0])
    for direction, label, color in ((normal[0], r'$\mathbf{n}$', MARKER),
                                     (binormal[0], r'$\mathbf{b}$', BINORMAL),
                                     (tangent[0], r'$\mathbf{T}$', INK)):
        end = project(centre[0] + .77 * direction)
        arrow(ax, base, end, color, width=1.1, scale=9, zorder=8)
        ax.text(*(base + 1.17 * (end - base)), label, fontsize=14, color=color,
                ha='center', va='center', zorder=9)
    ax.text(-.92, -.04, r'$s_0$', fontsize=14, color=MUTED)
    ax.text(.99, 2.33, r'$s_0+\ell$', fontsize=14, color=MUTED)
    ax.scatter(*base, color=INK, s=18, zorder=9)
    ax.set(xlim=(-1.03, 1.55), ylim=(-.3, 2.95), aspect='equal')
    ax.axis('off')


def angular_arrow(ax, phi, value, color, width=1.5):
    span = .25 * value
    if abs(span) < .025:
        return
    angles = np.linspace(phi, phi + span, 30)
    points = 1.1 * np.column_stack((np.cos(angles), np.sin(angles)))
    ax.plot(*points.T, color=color, linewidth=width, zorder=6)
    arrow(ax, points[-8], points[-1], color, width=width, scale=11, zorder=7)


def draw_section(ax):
    angles = np.linspace(0., 2 * np.pi, 361)
    ax.plot(np.cos(angles), np.sin(angles), color='#b6c2cd',
            linewidth=1.1, linestyle=(0, (3, 3)))
    arrow(ax, [-1.3, 0.], [1.50, 0.], MUTED, width=.8, scale=8)
    arrow(ax, [0., -1.3], [0., 1.5], MUTED, width=.8, scale=8)
    ax.text(1.57, -.06, r'$\mathbf{n}$', color=MARKER, fontsize=15)
    ax.text(.08, 1.46, r'$\mathbf{b}$', color=BINORMAL, fontsize=15)
    for phi in np.linspace(0., 2 * np.pi, 16, endpoint=False):
        if np.isclose(phi, 0.) or np.isclose(phi, np.pi / 2):
            continue
        angular_arrow(ax, phi, omega(phi), '#83a8c9', width=1.3)
    angular_arrow(ax, 0., omega(0.), MARKER, width=2.6)
    angular_arrow(ax, np.pi / 2, omega(np.pi / 2), BINORMAL, width=2.6)
    for point, color in (([1., 0.], MARKER), ([0., 1.], BINORMAL)):
        ax.scatter(*point, color=color, edgecolors='white', s=60, linewidths=.8, zorder=9)
    direction = np.array([np.cos(PHI), np.sin(PHI)])
    arrow(ax, [0., 0.], direction, RATE, width=2., scale=12, zorder=8)
    angle_arc = np.linspace(0., PHI, 40)
    ax.plot(.38 * np.cos(angle_arc), .38 * np.sin(angle_arc), color=RATE, linewidth=1.4)
    ax.text(.42, .08, r'$\phi$', fontsize=17, color=RATE)
    ax.text(.05, .57, r'$\mathbf{e}(\phi)$', fontsize=17, color=RATE)
    ax.text(1.12, -.40, r'$\Omega_n=p$', fontsize=16, color=MARKER, ha='center')
    ax.text(-.78, 1.25, r'$\Omega_b=-q$', fontsize=16, color=BINORMAL, ha='center')
    ax.scatter([0.], [0.], s=28, color=INK, zorder=9)
    ax.text(-.22, -.27, r'$\odot\ \mathbf{T}$', fontsize=15, color=INK, ha='right')
    ax.set(xlim=(-1.58, 1.77), ylim=(-1.35, 1.68), aspect='equal')
    ax.axis('off')


def draw_rates(ax):
    phi = np.linspace(0., 2 * np.pi, 721)
    rates = omega(phi)
    ax.fill_between(phi, ALPHA / 2, rates, color=RATE, alpha=.10)
    ax.axhline(0., color='#b6c2cd', linewidth=.9)
    ax.axhline(ALPHA / 2, color=MEAN, linewidth=1.5, linestyle=(0, (5, 3)))
    ax.plot(phi, rates, color=RATE, linewidth=2.6)
    ax.scatter([0., np.pi / 2], [omega(0.), omega(np.pi / 2)],
                color=[MARKER, BINORMAL], s=55, zorder=5, clip_on=False)
    ax.vlines(np.pi / 2, -.25, ALPHA / 2, color=BINORMAL, linewidth=.8, linestyle=':')
    ax.annotate(r'$\Omega_n L_0=pL_0$', (0., omega(0.)), (.22, 1.47),
                color=MARKER, fontsize=14, arrowprops=dict(arrowstyle='-', color=MARKER, lw=.8))
    ax.annotate(r'$\Omega_b L_0=-qL_0$', (np.pi / 2, omega(np.pi / 2)), (2.0, -.46),
                color=BINORMAL, fontsize=14, arrowprops=dict(arrowstyle='-', color=BINORMAL, lw=.8))
    ax.text(6.2, .56, r'$\langle\Omega\rangle_\phi L_0=\alpha L_0/2$', fontsize=14,
            color=MEAN, ha='right')
    ax.set(xlim=(0., 2 * np.pi), ylim=(-.57, 1.70))
    ax.set_xticks(np.arange(5) * np.pi / 2,
                  [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks([-.25, .0, .5, 1., 1.25])
    ax.tick_params(axis='both', colors=MUTED, labelsize=11, length=3)
    ax.set_ylabel(r'$\Omega L_0$', fontsize=16, color=INK, rotation=0, labelpad=26)
    ax.yaxis.set_label_coords(-.065, 1.04)
    ax.set_xlabel(r'$\phi$', fontsize=16, color=INK, labelpad=2)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ('bottom', 'left'))
        spine.set_color('#bbc6d0')


def make_figure(language, cjk_font=None):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    ja = language == 'ja'
    font = None
    if ja:
        path = cjk_font or Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
        if not path.exists():
            raise RuntimeError('Japanese rendering needs a CJK font; use --cjk-font.')
        fontManager.addfont(str(path))
        font = FontProperties(family=['DejaVu Sans', FontProperties(fname=str(path)).get_name()])
    fig = plt.figure(figsize=(18, 10.125), facecolor='white')

    def label(x, y, en, jp=None, size=12, color=INK, ha='left', **kwargs):
        if ja and jp:
            kwargs.pop('weight', None)
        return fig.text(x, y, jp if ja and jp else en, fontsize=size, color=color,
                        fontproperties=font if ja and jp else None, ha=ha, **kwargs)

    label(.04, .949, 'How neighbouring field lines turn: direction and azimuthal mean',
          '隣接磁力線の局所回転：方向依存と方位角平均', size=25, weight='bold')
    label(.04, .907, 'Rotation per unit arc length, measured in a transported normal plane',
          '基準磁力線のまわりの、単位弧長当たりの回転を考える', size=12.5, color=MUTED)
    fig.text(.96, .907, r'$\mathbf{T}=\mathbf{B}/B,\quad\mathbf{b}=\mathbf{T}\times\mathbf{n}$',
             fontsize=14, color=INK, ha='right')
    fig.text(.50, .85,
             r'$\Omega(\phi)=\mathbf{T}\cdot[\mathbf{e}\times((\mathbf{e}\cdot\nabla)\mathbf{T})]'
             r'=\frac{\alpha}{2}+\frac{\beta_g}{2}\cos2\phi-\frac{\delta_g}{2}\sin2\phi$',
             color=INK, fontsize=23, ha='center')
    headings = [(.04, .29, INK, '1  NEIGHBOURING FIELD LINES', '1  隣接する磁力線'),
                (.33, .60, RATE, '2  DIRECTIONAL ANGULAR RATES', '2  横断面での回転方向'),
                (.66, .96, MEAN, '3  UNIFORM AZIMUTHAL MEAN', '3  方位角について一様平均')]
    for left, right, color, en, jp in headings:
        fig.add_artist(Line2D([left, right], [.803, .803], color=color,
                              linewidth=2.5, transform=fig.transFigure))
        label((left + right) / 2, .773, en, jp, color=color, size=12, ha='center')
    fig.text(.108, .731, r'$M=$', fontsize=19, color=INK, va='center')
    matrix(fig, .192, .731, [[r'$a$', r'$q$'], [r'$p$', r'$d$']],
           size=15, width=.060, height=.044)
    draw_tube(fig.add_axes([.03, .343, .27, .355]))
    draw_section(fig.add_axes([.325, .387, .285, .326]))
    draw_rates(fig.add_axes([.682, .419, .278, .284]))
    fig.text(.465, .727, r'$\mathbf{e}(\phi)=\cos\phi\,\mathbf{n}+\sin\phi\,\mathbf{b}$',
             fontsize=16, color=INK, ha='center')
    fig.text(.465, .354, r'$\Omega_n=\Omega(0),\qquad\Omega_b=\Omega(\pi/2)$',
             fontsize=15, color=INK, ha='center')
    label(.165, .314, 'Dark: reference   Colour: neighbours',
          '濃色：基準磁力線　橙・緑：同じ隣接磁力線を追う', size=10.5, color=MUTED, ha='center')
    label(.465, .314, 'T out of page; positive: n to b',
          'T は手前向き。正の回転は n から b へ', size=10.5, color=MUTED, ha='center')
    label(.81, .342, 'A positive mean can include negative directions.',
          '平均が正でも、逆向きに回る方向がある。', size=11.5, color=INK, ha='center')
    label(.81, .314, 'Shading: directional departure from the mean',
          '青い帯：平均からの方向ごとのずれ', size=10.5, color=MUTED, ha='center')
    label(.165, .288,
          r'$\ell=0.9L_0$: arc length between sections along the reference line',
          'ℓ = 0.9L₀：両断面間の基準磁力線に沿った弧長',
          size=9.5, color=INK, ha='center')
    label(.465, .288, 'Circle: directions at s0, not a finite mapped boundary',
          '円は s₀ での方向を示す（変形後の管の輪郭ではない）', size=9.5, color=MUTED, ha='center')
    label(.81, .288,
          r'$L_0$: chosen reference length; $\Omega L_0$: dimensionless',
          'L₀：任意に選んだ基準長。ΩL₀ は無次元。',
          size=10, color=INK, ha='center')

    for left, width, color, title, jp in (
            (.04, .245, MARKER, 'SUM  /  FAC', '和：FAC に対応'),
            (.305, .245, BINORMAL, 'DIFFERENCE  /  SIGNED SHEAR', '差：符号付きせん断'),
            (.57, .39, MEAN, 'MEAN  /  HALF OF ALPHA', '一様平均：α の半分')):
        fig.add_artist(FancyBboxPatch((left, .124), width, .139,
                                      boxstyle='round,pad=0.006,rounding_size=0.006',
                                      transform=fig.transFigure, linewidth=0,
                                      facecolor=color, alpha=.055, zorder=-1))
        label(left + .013, .235, title, jp, size=11, color=color)
    fig.text(.1625, .181, r'$\Omega_n+\Omega_b=\alpha=\frac{\mu_0j_\parallel}{B}$',
             fontsize=19, color=INK, ha='center')
    fig.text(.4275, .181, r'$\Omega_n-\Omega_b=\beta_g=\frac{\mathcal{D}}{B}$',
             fontsize=19, color=INK, ha='center')
    fig.text(.765, .180, r'$\langle\Omega\rangle_\phi=\frac{1}{2\pi}\int_0^{2\pi}\Omega(\phi)\,d\phi'
             r'=\frac{\alpha}{2}=\frac{\mu_0j_\parallel}{2B}$',
             fontsize=19, color=MEAN, ha='center')
    label(.1625, .14, 'p + (-q)', 'p + (−q)', size=11, color=MUTED, ha='center')
    label(.4275, .14, 'p - (-q)', 'p − (−q)', size=11, color=MUTED, ha='center')
    label(.765, .14, 'Uniform directions at one point; not an along-line average',
          '同じ点での方位角平均。磁力線に沿った平均とは異なる。', size=10, color=MUTED, ha='center')
    label(.04, .084, 'Illustration:', '図の例：', size=10.5, color=MUTED)
    fig.text(.11, .084, r'$(\alpha,\beta_g,\delta_g)=(1,1.5,0.5)/L_0,\quad\theta=0$',
             fontsize=12, color=MUTED)
    label(.96, .084, 'Arrows show instantaneous angular motion; radial deformation is omitted in panel 2.',
          '矢印は局所的な角度変化を示す。中央図では半径方向の変形を省略。',
          size=10, color=MUTED, ha='right')
    fig.add_artist(Line2D([.04, .96], [.065, .065], color='#d8e0e7', lw=.8, transform=fig.transFigure))
    label(.04, .034, 'Local thin-tube model; the planar reference line has zero torsion.',
          '局所的な細い管の模式図。基準磁力線は平面曲線で、ねじれ率 τ = 0。',
          size=10.5, color=MUTED)
    label(.96, .034, 'In a general Frenet frame:', '一般の Frenet 基底では：',
          size=10.5, color=MUTED, ha='right')
    # Leave a separate line for the frame correction so it is explicit in both languages.
    fig.text(.96, .009, r'$d\phi/ds=\Omega(\phi)-\tau$', fontsize=12, color=INK, ha='right')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/images')
    parser.add_argument('--language', choices=('en', 'ja', 'both'), default='en')
    parser.add_argument('--cjk-font', type=Path)
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
        fig = make_figure(language, args.cjk_font)
        stem = 'directional-rotation' + ('-ja' if language == 'ja' else '')
        for extension in ('svg', 'pdf', 'png'):
            path = args.output_dir / f'{stem}.{extension}'
            fig.savefig(path, dpi=180, facecolor='white')
            if extension == 'svg':
                lines = path.read_text(encoding='utf-8').splitlines()
                path.write_text('\n'.join(line.rstrip() for line in lines) + '\n', encoding='utf-8')
            print(f'Saved {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
