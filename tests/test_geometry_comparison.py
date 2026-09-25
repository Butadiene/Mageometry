"""Comparison contracts and real VTK case-selection regressions."""

import gc
import io
import sys
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, viz3d
from mageometry.viz3d._dropdown import _Dropdown
from mageometry.viz3d._overview_data import _OverviewData, _validate_cases
from mageometry.viz3d.fac import _peak_projection
from mageometry.viz3d.slicer import _widget_state

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def grid(strength=1., axis=None, metadata=None):
    axis = np.linspace(-2, 2, 13) if axis is None else axis
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return GriddedField(axis, axis, axis, -strength*y*z, strength*x*z,
                        np.ones_like(x), metadata=metadata)


def nonlinear_field(strength):
    def field(x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        return -strength*np.sin(y), strength*np.sin(x), np.ones_like(z)
    return field


def model_grid(field, count=5):
    axis = np.linspace(-2, 2, count)
    return GriddedField(axis, axis, axis,
                        *field(*np.meshgrid(axis, axis, axis, indexing='ij')))


def press(plotter, key):
    plotter.iren.interactor.SetKeyCode(key if len(key) == 1 else '\0')
    plotter.iren.interactor.SetKeySym(key)
    plotter.iren.interactor.InvokeEvent('KeyPressEvent')
    plotter.iren.interactor.InvokeEvent('CharEvent')


def click(plotter, name):
    actor = plotter.renderers[0].actors[name]
    x, y = actor.GetPositionCoordinate().GetComputedDisplayValue(plotter.renderers[0])
    plotter.iren.interactor.SetEventPosition(x + 3, y)
    plotter.iren.interactor.InvokeEvent('LeftButtonPressEvent')
    plotter.iren.interactor.InvokeEvent('LeftButtonReleaseEvent')


def select_case(plotter, label):
    click(plotter, 'geometry-dataset-value')
    click(plotter, f'geometry-dataset-option-{label}')


def camera_state(renderer):
    camera = renderer.camera
    return np.r_[camera.position, camera.focal_point, camera.up,
                 camera.parallel_scale, camera.parallel_projection,
                 camera.GetWindowCenter(), camera.clipping_range]


class TestComparisonData(unittest.TestCase):

    def test_shear_values_are_unscaled_with_explicit_color_limits(self):
        data = _OverviewData({'A': grid(), 'B': grid(-2.)}, comparison=True,
                             current_scale=0.125,
                             color_limits={'beta_g': .75, 'delta_g': .5})
        for name, limit in (('beta_g', .75), ('delta_g', .5)):
            with self.subTest(name=name):
                selection = data.prepare('A', name)
                self.assertEqual(selection.component, name)
                self.assertEqual(selection.scale.limit, limit)
                native, basis = selection.prepared.cache.get(name)
                self.assertIsNone(basis)
                np.testing.assert_allclose(selection.values, native)

    @unittest.skipUnless(HAVE_PV, 'pyvista not installed')
    def test_comparison_rejects_removed_names_and_color_limit_keys(self):
        for name in ('sigma', 'q'):
            for options in ({'component': name}, {'color_limits': {name: 1.}}):
                with self.subTest(name=name, options=options), \
                        self.assertRaisesRegex(ValueError, 'Unknown current component'):
                    viz3d.compare_geometry({'A': grid()}, show=False, **options)

    def test_direct_fields_require_complete_mapping_and_explicit_step(self):
        cases = {'A': grid(), 'B': grid()}
        fields = dict.fromkeys(cases, nonlinear_field(1.))
        invalid = (
            {'fields': []}, {'fields': {}}, {'fields': {'A': fields['A']}},
            {'fields': dict(fields, C=fields['A'])},
            {'fields': dict(fields, B=None)}, {'fields': fields, 'delta': None},
        )
        invalid += tuple({'fields': fields, 'delta': step}
                         for step in (0, -1, np.nan, np.inf, [0.1, 0.2], [0.1, 0., 0.2]))
        for options in invalid:
            options = dict({'delta': 0.002}, **options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                viz3d.compare_geometry(cases, show=False, **options)

    def test_direct_derivatives_are_independent_of_display_spacing(self):
        fields = {'A': nonlinear_field(1.), 'B': nonlinear_field(-2.)}
        delta = 0.002
        for count in (5, 9):
            cases = {label: model_grid(field, count) for label, field in fields.items()}
            data = _OverviewData(cases, comparison=True, fields=fields,
                                 delta=delta, cache_size=1)
            centre = (count // 2,) * 3
            for label, strength in (('A', 1.), ('B', -2.), ('A', 1.)):
                for component in ('alpha', 'fac'):
                    result = data.prepare(label, component)
                    self.assertEqual(result.prepared.cache.delta, delta)
                    # At the origin B=(0,0,1), curl(B).z=2*s*sin(h)/h.
                    self.assertAlmostEqual(result.values[centre],
                                           2*strength*np.sin(delta)/delta, places=11)
            grid_only = _OverviewData(cases, comparison=True).prepare('A', 'alpha')
            self.assertGreater(abs(grid_only.values[centre] - 2*np.sin(delta)/delta), 0.01)

    def test_geometry_step_override_keeps_cartesian_fac_step(self):
        field = nonlinear_field(1.)
        data = _OverviewData({'A': model_grid(field)}, comparison=True,
                             fields={'A': field}, delta=(0.1, 0.2, 0.3),
                             geometry_delta=0.002)
        centre = (2, 2, 2)
        self.assertAlmostEqual(data.prepare('A', 'fac').values[centre],
                               np.sin(0.1)/0.1 + np.sin(0.2)/0.2)
        self.assertAlmostEqual(data.prepare('A', 'alpha').values[centre],
                               2*np.sin(0.002)/0.002)
        default = _OverviewData({'A': model_grid(field)}, comparison=True,
                                fields={'A': field}, delta=(0.1, 0.2, 0.3))
        self.assertAlmostEqual(default.prepare('A', 'alpha').values[centre],
                               2*np.sin(0.1)/0.1)

    def test_direct_fields_preserve_grid_nans_and_skip_masked_points(self):
        model = nonlinear_field(1.)
        case = model_grid(model)
        case.b[1, 1, 1] = np.nan

        def field(x, y, z):
            self.assertTrue(np.all(x*x + y*y + z*z >= 0.25**2))
            return model(x, y, z)

        data = _OverviewData({'A': case}, comparison=True, fields={'A': field},
                             delta=0.002, mask=lambda x, y, z: x*x+y*y+z*z < 0.25**2)
        with patch.object(GriddedField, 'field', side_effect=AssertionError('interpolation used')):
            for component in ('fac', 'alpha'):
                values = data.prepare('A', component).values
                self.assertTrue(np.isnan(values[1, 1, 1]))
                self.assertTrue(np.isnan(values[2, 2, 2]))
                self.assertTrue(np.isfinite(values[0, 0, 0]))

    def test_axes_and_declared_units_are_validated(self):
        for cases in ({}, [], {'': grid()}, {'A': object()}):
            with self.subTest(cases=type(cases)), self.assertRaises((ValueError, TypeError)):
                _validate_cases(cases)
        with self.assertRaisesRegex(ValueError, 'different axes'):
            _validate_cases({'A': grid(), 'B': grid(axis=np.linspace(-3, 3, 13))})
        for key in ('coordinate_system', 'length_unit', 'field_unit'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                _validate_cases({'A': grid(metadata={key: 'unit1'}),
                                 'B': grid(metadata={key: 'unit2'})})
        self.assertEqual(len(_validate_cases({'A': grid(), 'B': grid(metadata={'field_unit': 'nT'})})), 2)

    def test_shared_statistics_reference_threshold_and_reproducibility(self):
        cases = {'A': grid(), 'B': grid(-3.)}
        data = _OverviewData(cases, comparison=True, cache_size=1, percentile=80)
        a = data.prepare('A', 'fac')
        b = data.prepare('B', 'fac')
        va = np.abs(a.values[np.isfinite(a.values)])
        vb = np.abs(b.values[np.isfinite(b.values)])
        self.assertEqual(a.scale, b.scale)
        self.assertEqual(a.scale.limit, max(np.percentile(va, 98), np.percentile(vb, 98)))
        self.assertEqual(a.scale.peak, max(va.max(), vb.max()))
        self.assertEqual(a.scale.threshold, np.percentile(va, 80))
        self.assertFalse(np.allclose(a.values, b.values, equal_nan=True))
        np.testing.assert_allclose(data.prepare('A', 'fac').values, a.values)

    def test_fixed_limits_units_and_invalid_options(self):
        cases = {'A': grid()}
        native = _OverviewData(cases, comparison=True)
        scaled = _OverviewData(cases, comparison=True, current_scale=2,
                                color_limits={'fac': 0.25, 'alpha': 0.75})
        for key in ('fac', 'alpha'):
            a, b = native.prepare('A', key), scaled.prepare('A', key)
            np.testing.assert_allclose(b.values, a.values * (2 if key == 'fac' else 1))
            self.assertEqual(b.scale.limit, 0.25 if key == 'fac' else 0.75)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                _OverviewData(cases, cache_size=size)
        for limits in ([], {'unknown': 1}, {'fac': 0}, {'fac': np.nan}, {'fac': -1}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                _OverviewData(cases, color_limits=limits)
        for options in ({'field': lambda *args: None}, {'delta': 0.1},
                        {'initial_case': 'missing'}, {'component': None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                viz3d.compare_geometry(cases, show=False, **options)

    def test_empty_and_zero_cases_have_finite_shared_scales(self):
        empty = grid()
        empty.b[:] = np.nan
        for cases in ({'empty': empty}, {'zero': grid(0)}, {'empty': empty, 'valid': grid()}):
            data = _OverviewData(cases, comparison=True)
            result = data.prepare(next(iter(cases)), 'fac')
            self.assertTrue(np.isfinite(result.scale.limit))
            self.assertGreater(result.scale.limit, 0)
            self.assertEqual(result.scale.threshold, 0)
        self.assertTrue(np.all(np.isnan(_OverviewData({'empty': empty}).prepare('empty', 'alpha').values)))

    def test_eviction_releases_previews_and_failure_keeps_selection(self):
        data = _OverviewData({'A': grid(), 'B': grid(-1)}, comparison=True, cache_size=1)
        old = data.get('A')
        ref = weakref.ref(old)
        del old
        data.get('B')
        gc.collect()
        self.assertIsNone(ref())
        selection = data.prepare('A', 'fac')
        data.commit(selection)
        with patch.object(data, 'get', side_effect=RuntimeError('sampling failed')):
            with self.assertRaises(RuntimeError):
                data.prepare('B', 'fac')
        self.assertIs(data.selection, selection)
        self.assertLessEqual(len(data.cache), 1)


class TestT96ComparisonExample(unittest.TestCase):
    def test_direct_cases_keep_parameters_and_epoch_after_other_evaluations(self):
        from mageometry import geopack, geopack_field
        from examples.compare_t96_by import make_fields, EPOCH, case_label
        by_values = np.array([-5., 5.])
        fields = make_fields(by_values)
        points = (np.array([[-6.], [-10.]]), np.array([0.3, 0.7]), 1.)
        expected = {}
        for by in by_values:
            ps = geopack.recalc(EPOCH)
            model = geopack_field('t96', 'dip', [2., -20., by, -5., 0, 0, 0, 0, 0, 0], ps)
            expected[case_label(by)] = model(*points)
        self.assertFalse(np.allclose(*expected.values()))
        by_values[:] = 100.
        for label in (case_label(-5), case_label(5), case_label(-5)):
            geopack.recalc(1609459200.)
            np.testing.assert_allclose(fields[label](*points), expected[label], rtol=1e-13)
            self.assertTrue(all(np.isscalar(v) for v in fields[label](-6., 0.3, 1.)))

    def test_direct_diagnostic_matches_original_model_after_case_eviction(self):
        from mageometry import geopack, geopack_field, field_line_transverse_geometry
        from examples.compare_t96_by import make_cases, make_fields, EPOCH, MODEL_DELTA, inner_mask
        cases = make_cases([-5., 5.], shape=(9, 7, 7))
        fields = make_fields([-5., 5.])
        data = _OverviewData(cases, comparison=True, fields=fields,
                             delta=MODEL_DELTA, mask=inner_mask, cache_size=1)
        for label in (*cases, next(iter(cases))):
            case = cases[label]
            ps = geopack.recalc(EPOCH)
            model = geopack_field('t96', 'dip',
                                  [2., -20., case.metadata['imf_by'], -5., 0, 0, 0, 0, 0, 0], ps)
            expected = field_line_transverse_geometry(
                model, case.x[3], case.y[4], case.z[4], delta=0.002)
            geopack.recalc(1609459200.)
            result = data.prepare(label, 'alpha')
            self.assertAlmostEqual(result.values[3, 4, 4], expected['alpha'], places=12)
            self.assertEqual(result.prepared.cache.delta, 0.002)

    def test_cli_defaults_and_grid_opt_in(self):
        from examples import compare_t96_by as example
        runs = (
            ([], True, 0.002, None),
            (['--component', 'delta_g'], True, 0.002, None),
            (['--evaluation', 'grid'], False, None, None),
            (['--delta', '0.001', '--geometry-delta', '0.003'], True, 0.001, 0.003),
            (['--evaluation', 'grid', '--geometry-delta', '0.2'], False, None, 0.2),
        )
        for args, direct, delta, geometry_delta in runs:
            with self.subTest(args=args), patch.object(sys, 'argv', ['compare', *args]), \
                    patch.object(example, 'make_cases', return_value={'A': grid()}) as cases, \
                    patch.object(example, 'make_fields', return_value={'A': nonlinear_field(1.)}) as fields, \
                    patch.object(example.viz3d, 'compare_geometry') as render, patch('builtins.print'):
                example.main()
                cases.assert_called_once_with(example.DEFAULT_BY, (65, 49, 49))
                self.assertEqual(render.call_args.kwargs['delta'], delta)
                self.assertEqual(render.call_args.kwargs['geometry_delta'], geometry_delta)
                self.assertEqual(render.call_args.kwargs['component'],
                                 'delta_g' if '--component' in args else 'alpha')
                if direct:
                    self.assertIs(render.call_args.kwargs['fields'], fields.return_value)
                else:
                    fields.assert_not_called()
                    self.assertIsNone(render.call_args.kwargs['fields'])

    def test_cli_rejects_invalid_steps_and_removed_names_before_sampling(self):
        from examples import compare_t96_by as example
        runs = (['--evaluation', 'grid', '--delta', '0.002'], ['--delta', '0'],
                ['--geometry-delta', 'nan'], ['--delta', 'inf'],
                ['--component', 'sigma'], ['--component', 'q'])
        for args in runs:
            with self.subTest(args=args), patch.object(sys, 'argv', ['compare', *args]), \
                    patch.object(example, 'make_cases') as sample, \
                    patch.object(sys, 'stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    example.main()
                self.assertEqual(error.exception.code, 2)
                sample.assert_not_called()

    def test_independent_six_case_snapshots_and_fixed_conditions(self):
        from examples.compare_t96_by import make_cases, DEFAULT_BY
        cases = make_cases(shape=(9, 7, 7))
        self.assertEqual(len(cases), 6)
        first = next(iter(cases.values()))
        for by, case in zip(DEFAULT_BY, cases.values()):
            self.assertEqual(case.metadata['imf_by'], by)
            parameters = case.metadata['parameters']
            for label, key in (('IMF By [nT]', 'imf_by'), ('IMF Bz [nT]', 'imf_bz'),
                               ('Pdyn [nPa]', 'pdyn'), ('Dst [nT]', 'dst'),
                               ('Dipole tilt [rad]', 'dipole_tilt'), ('Epoch [Unix s]', 'epoch')):
                self.assertEqual(parameters[label], case.metadata[key])
            for key in ('epoch', 'dipole_tilt', 'pdyn', 'dst', 'imf_bz'):
                self.assertEqual(case.metadata[key], first.metadata[key])
            np.testing.assert_array_equal(case.x, first.x)
            if case is not first:
                self.assertFalse(np.shares_memory(first.b, case.b))
                self.assertFalse(np.allclose(first.b, case.b, equal_nan=True))
        before = first.b.copy()
        make_cases([7.], shape=(9, 7, 7))
        np.testing.assert_array_equal(first.b, before)

    def test_invalid_scan_inputs(self):
        from examples.compare_t96_by import make_cases
        for values in ([], [np.nan], [1, 1], [[1, 2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_cases(values)
        with self.assertRaises(ValueError):
            make_cases(shape=(2, 7, 7))


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestComparisonViewer(unittest.TestCase):
    def tearDown(self):
        pv.close_all()

    def viewer(self, **options):
        defaults = dict(component='fac', threshold=0.25, slice_panel=True,
                        slice_normal='z', slice_origin=(0, 0, 0.5), n_lines=0, show=False)
        defaults.update(options)
        p = viz3d.compare_geometry({'A': grid(), 'B': grid(-2.)}, **defaults)
        p.screenshot()
        return p

    def test_switch_preserves_cameras_slice_cutoff_and_common_scale(self):
        p = self.viewer()
        widget = _widget_state(p).plane_widgets[-1]
        widget.SetOrigin(0, 0, 0.75)
        widget.InvokeEvent('InteractionEvent')
        cameras = [camera_state(r) for r in p.renderers]
        cutoff = _widget_state(p).slider_widgets[0].GetRepresentation()
        first = p.actors['fac-slice'].mapper.dataset['fac'].copy()
        limits = p.actors['fac-slice'].mapper.scalar_range
        cutoff.SetValue(0.6)
        _widget_state(p).slider_widgets[0].InvokeEvent('EndInteractionEvent')
        select_case(p, 'B')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'B')
        self.assertEqual(cutoff.GetValue(), 0.6)
        self.assertEqual(p.actors['fac-slice'].mapper.scalar_range, limits)
        self.assertFalse(np.allclose(p.actors['fac-slice'].mapper.dataset['fac'], first))
        np.testing.assert_allclose(p.actors['fac-slice'].mapper.dataset.points[:, 2], 0.75)
        values = _OverviewData({'B': grid(-2.)}).prepare('B', 'fac').values
        for axis, renderer in enumerate(p.renderers[1:4]):
            actor = renderer.actors['fac-projection']
            projection = _peak_projection(values, axis).ravel(order='F')
            np.testing.assert_allclose(actor.mapper.dataset['fac'],
                                       np.where(np.abs(projection) >= 0.6, projection, np.nan))
            self.assertEqual(actor.mapper.scalar_range, limits)
        for renderer, camera in zip(p.renderers, cameras):
            np.testing.assert_allclose(camera_state(renderer), camera)
        select_case(p, 'A')
        np.testing.assert_allclose(p.actors['fac-slice'].mapper.dataset['fac'], first)
        self.assertEqual(cutoff.GetValue(), 0.6)
        self.assertIn('A\nfac', p.renderers[4].actors['fac-panel-title'].GetInput())
        press(p, 'F6')
        select_case(p, 'B')
        press(p, 'F5')
        self.assertEqual(cutoff.GetValue(), 0.6)

    def test_initial_case_keeps_reference_threshold_and_automatic_seeds(self):
        from mageometry.tracing import trace_field_lines
        calls = []
        for initial_case in ('A', 'B'):
            with patch('mageometry.viz3d.fac.trace_field_lines', wraps=trace_field_lines) as trace:
                p = self.viewer(initial_case=initial_case, threshold=None, n_lines=3,
                                trace_kwargs={'max_steps': 5, 'ds': 0.05})
                calls.append(np.array(trace.call_args.args[1:]).T.copy())
            expected = _OverviewData({'A': grid()}).prepare('A', 'fac').scale.threshold
            self.assertEqual(_widget_state(p).slider_widgets[0].GetRepresentation().GetValue(), expected)
            self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), initial_case)
            p.close()
        self.assertGreater(len(calls[0]), 0)
        np.testing.assert_array_equal(calls[0], calls[1])

    def test_linked_menus_and_keyboard_work_in_enlarged_slice(self):
        p = self.viewer()
        press(p, 'F4')
        camera = camera_state(p.renderers[0])
        click(p, 'current-component-value')
        click(p, 'geometry-dataset-value')
        self.assertFalse(p.actors['current-component-option-alpha'].GetVisibility())
        self.assertTrue(p.actors['geometry-dataset-option-B'].GetVisibility())
        press(p, 'End')
        press(p, 'Return')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'B')
        click(p, 'geometry-dataset-value')
        click(p, 'current-component-value')
        click(p, 'current-component-option-alpha')
        self.assertEqual(p.actors['fac-slice'].mapper.array_name, 'alpha')
        limits = p.actors['fac-slice'].mapper.scalar_range
        press(p, 'F7')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'A')
        self.assertEqual(p.actors['fac-slice'].mapper.array_name, 'alpha')
        self.assertEqual(p.actors['fac-slice'].mapper.scalar_range, limits)
        np.testing.assert_allclose(camera_state(p.renderers[0]), camera)
        self.assertFalse(p.renderers[4].GetDraw())
        press(p, 'F4')
        self.assertTrue(p.renderers[4].GetDraw())

    def test_lines_are_retraced_from_fixed_seeds_and_keep_visibility(self):
        from mageometry.tracing import trace_field_lines
        seeds = np.array([[0.5, 0., 0.5], [-0.5, 0., 0.5]])
        with patch('mageometry.viz3d.fac.trace_field_lines', wraps=trace_field_lines) as trace:
            p = self.viewer(seeds=seeds, trace_kwargs={'max_steps': 20, 'ds': 0.05})
            first = p.actors['fac-lines'].mapper.dataset.points.copy()
            press(p, 'l')
            select_case(p, 'B')
            self.assertFalse(p.actors['fac-lines'].visibility)
            self.assertFalse(np.allclose(first, p.actors['fac-lines'].mapper.dataset.points))
            press(p, 'F6')
            self.assertEqual(trace.call_count, 2)  # diagnostic changes do not retrace
            select_case(p, 'A')
            np.testing.assert_allclose(p.actors['fac-lines'].mapper.dataset.points, first)
            for call in trace.call_args_list:
                np.testing.assert_array_equal(np.array(call.args[1:]).T, seeds)

    def test_direct_case_selection_uses_model_for_diagnostics_and_tracing(self):
        from mageometry.tracing import trace_field_lines
        fields = {'A': nonlinear_field(1.), 'B': nonlinear_field(-2.)}
        cases = {label: model_grid(field) for label, field in fields.items()}
        seeds = np.array([[0.3, 0.4, 0.5]])
        with patch('mageometry.viz3d.fac.trace_field_lines', wraps=trace_field_lines) as trace:
            p = viz3d.compare_geometry(cases, fields=fields, delta=0.002,
                                       component='alpha', seeds=seeds, cache_size=1,
                                       trace_kwargs={'ds': 0.05, 'max_steps': 10},
                                       slice_panel=True, slice_normal='z', show=False)
            first = p.actors['fac-lines'].mapper.dataset.points.copy()
            cameras = [camera_state(r) for r in p.renderers]
            limits = p.actors['fac-slice'].mapper.scalar_range
            select_case(p, 'B')
            self.assertFalse(np.allclose(first, p.actors['fac-lines'].mapper.dataset.points))
            np.testing.assert_allclose(trace.call_args.args[0](*seeds.T), fields['B'](*seeds.T))
            self.assertEqual(p.actors['fac-slice'].mapper.scalar_range, limits)
            select_case(p, 'A')
            np.testing.assert_allclose(p.actors['fac-lines'].mapper.dataset.points, first)
            np.testing.assert_allclose(trace.call_args.args[0](*seeds.T), fields['A'](*seeds.T))
            for renderer, camera in zip(p.renderers, cameras):
                np.testing.assert_allclose(camera_state(renderer), camera)
            values = p.actors['fac-slice'].mapper.dataset['alpha']
            self.assertTrue(np.any(np.isclose(values, 2*np.sin(0.002)/0.002)))

    def test_failed_calculation_keeps_rendered_case_and_label(self):
        menus = []

        def capture(*args, **kwargs):
            menu = _Dropdown(*args, **kwargs)
            menus.append(menu)
            return menu

        with patch('mageometry.viz3d.fac._Dropdown', side_effect=capture):
            p = self.viewer()
        before = p.actors['fac-slice'].mapper.dataset['fac'].copy()
        with patch.object(_OverviewData, 'prepare', side_effect=RuntimeError('sampling failed')):
            with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
                menus[-1].callback('B')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'A')
        self.assertNotIn('fac-focus-loading', p.actors)
        np.testing.assert_array_equal(p.actors['fac-slice'].mapper.dataset['fac'], before)
        select_case(p, 'B')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'B')

    def test_empty_case_does_not_change_scale_or_lower_threshold(self):
        empty = grid()
        empty.b[:] = np.nan
        p = viz3d.compare_geometry({'A': grid(), 'empty': empty}, component='fac',
                                   threshold=1, n_lines=0, slice_panel=True,
                                   slice_normal='z', slice_origin=(0, 0, 0.5), show=False)
        p.screenshot()
        limits = p.actors['fac-slice'].mapper.scalar_range
        select_case(p, 'empty')
        self.assertEqual(p.actors['geometry-dataset-value'].GetInput(), 'empty')
        self.assertIn('unavailable', p.actors['fac-status'].GetInput())
        self.assertFalse(p.actors['fac-slice'].visibility)
        self.assertFalse(p.renderers[4].actors['fac-panel-slice'].visibility)
        self.assertEqual(_widget_state(p).slider_widgets[0].GetRepresentation().GetValue(), 1)
        select_case(p, 'A')
        self.assertTrue(p.actors['fac-slice'].visibility)
        self.assertEqual(p.actors['fac-slice'].mapper.scalar_range, limits)


if __name__ == '__main__':
    unittest.main()
