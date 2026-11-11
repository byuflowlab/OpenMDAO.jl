"""
End-to-end tests for the implicit automatic-differentiation components
(`ADImplicitComp`) in OpenMDAOCore.jl, run through omjlcomps' `JuliaImplicitComp`
inside OpenMDAO `Problem`s.

The test component solves the implicit system

    R.z1[i] = a*x^2 + y[i]^2 - z1[i]
    R.z2    = a*x + sum(y) - z2

whose exact solution is

    z1[i] = a*x^2 + y[i]^2
    z2    = a*x + sum(y)

so the states, the partial derivatives of the residuals, and the total
derivatives of the states are all known analytically.
"""
import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaImplicitADCompTest")
import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from omjlcomps import JuliaImplicitComp

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_implicit_ad_comps.jl"))


def _expected_states(x, y, a):
    return a * x**2 + y**2, a * x + np.sum(y)


def _expected_residual_partials(x, y, a):
    """Analytic partial derivatives of the residuals wrt inputs and states."""
    n = y.size
    p = {}
    p["z1", "x"] = np.full((n, 1), 2.0 * a * x)
    p["z1", "y"] = np.diag(2.0 * y)
    p["z1", "z1"] = -np.eye(n)
    p["z1", "z2"] = np.zeros((n, 1))
    p["z2", "x"] = np.array([[a]])
    p["z2", "y"] = np.ones((1, n))
    p["z2", "z1"] = np.zeros((1, n))
    p["z2", "z2"] = -np.ones((1, 1))
    return p


def _expected_totals(x, y, a):
    """Analytic total derivatives of the solved states wrt the inputs."""
    n = y.size
    t = {}
    t["z1", "x"] = np.full((n, 1), 2.0 * a * x)
    t["z1", "y"] = np.diag(2.0 * y)
    t["z2", "x"] = np.array([[a]])
    t["z2", "y"] = np.ones((1, n))
    return t


def _check_partials_vs_analytic(p, comp_name, x, y, a, methods=("fd",)):
    expected = _expected_residual_partials(x, y, a)
    for method in methods:
        cpd = p.check_partials(compact_print=True, out_stream=None, method=method)
        comp_partials = cpd[comp_name]
        for (of, wrt), J_analytic in expected.items():
            J = comp_partials[of, wrt][f"J_{method}"]
            np.testing.assert_allclose(J, J_analytic, rtol=1e-6, atol=1e-8)


@use_tempdirs
class TestDenseImplicitADInPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_test_comp(n=n, a=a, in_place=True)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_dense_ad_in_place", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_partials_fd(self):
        _check_partials_vs_analytic(self.p, "icomp_dense_ad_in_place", self.x, self.y, self.a, methods=("fd",))

    def test_apply_nonlinear_complex_step(self):
        # The AD implicit comps keep ComplexF64 copies of the state/input
        # vectors so that OpenMDAO's complex-step method works. Verify directly
        # by calling apply_nonlinear with complex-perturbed inputs.
        comp = self.p.model.icomp_dense_ad_in_place
        h = 1e-30

        inputs = {"x": np.array(self.x + 1j * h), "y": self.y.astype(complex)}
        outputs = {"z1": np.zeros(self.n, dtype=complex), "z2": np.zeros(1, dtype=complex)}
        residuals = {"z1": np.zeros(self.n, dtype=complex), "z2": np.zeros(1, dtype=complex)}
        comp.apply_nonlinear(inputs, outputs, residuals)

        # dR_z1/dx = 2*a*x; dR_z2/dx = a
        np.testing.assert_allclose(np.imag(residuals["z1"]) / h, 2.0 * self.a * self.x * np.ones(self.n), rtol=1e-8)
        np.testing.assert_allclose(np.imag(residuals["z2"]) / h, self.a, rtol=1e-8)


@use_tempdirs
class TestDenseImplicitADOutOfPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_test_comp(n=n, a=a, in_place=False)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_dense_ad_out_of_place", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_partials_fd(self):
        _check_partials_vs_analytic(self.p, "icomp_dense_ad_out_of_place", self.x, self.y, self.a, methods=("fd",))


@use_tempdirs
class TestSparseImplicitAD(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_sparse_implicit_test_comp(n=n, a=a)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_sparse_ad", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_partials_fd(self):
        _check_partials_vs_analytic(self.p, "icomp_sparse_ad", self.x, self.y, self.a, methods=("fd",))


@use_tempdirs
class TestMatrixFreeForwardImplicitAD(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_matrix_free_implicit_test_comp(n=n, a=a, reverse=False)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=False)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_matrix_free_forward_ad", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_totals(self):
        p = self.p
        totals = p.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
        expected = _expected_totals(self.x, self.y, self.a)
        for (of, wrt), J_analytic in expected.items():
            np.testing.assert_allclose(totals[of, wrt], J_analytic, rtol=1e-6, atol=1e-8)


@use_tempdirs
class TestMatrixFreeReverseImplicitAD(unittest.TestCase):
    """
    The reverse matrix-free flavor only supports `apply_linear` with
    `mode="rev"`, so OpenMDAO's Newton solver (which always solves in "fwd"
    mode) cannot be used with it. Instead, this test verifies the problem
    setup/plumbing and the reverse (vector-Jacobian-product) path directly.
    """

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_matrix_free_implicit_test_comp(n=n, a=a, reverse=True)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("icomp_matrix_free_ad_reverse", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_apply_linear_rev(self):
        comp = self.p.model.icomp_matrix_free_ad_reverse
        n = self.n
        a = self.a
        x = self.x
        y = self.y

        inputs = {"x": np.array([x]), "y": y.copy()}
        outputs = {"z1": np.zeros(n), "z2": np.zeros(1)}
        d_inputs = {"x": np.zeros(1), "y": np.zeros(n)}
        d_outputs = {"z1": np.zeros(n), "z2": np.zeros(1)}
        d_residuals = {"z1": np.ones(n), "z2": np.zeros(1)}

        comp.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "rev")

        # VJP of the residuals wrt (x, y) with seed dR.z1 = ones(n) (and
        # dR.z2 = 0). The seed applies to every entry of R.z1, so
        #   dx = sum_i dR_z1[i]/dx * 1 = n * 2*a*x
        #   dy = dR_z1/dy * seed = 2*y  (diagonal)
        np.testing.assert_allclose(d_inputs["x"], np.array([n * 2.0 * a * x]), rtol=1e-10)
        np.testing.assert_allclose(d_inputs["y"], 2.0 * y, rtol=1e-10)
        # d_outputs gets the dR/dz contribution: -seed on z1, nothing on z2.
        np.testing.assert_allclose(d_outputs["z1"], -np.ones(n), rtol=1e-10)
        np.testing.assert_allclose(d_outputs["z2"], np.zeros(1), rtol=1e-10)


@use_tempdirs
class TestMatrixFreeForwardReverseImplicitAD(unittest.TestCase):
    """
    The forward+reverse matrix-free flavor supports both `apply_linear`
    directions, so unlike the single-direction matrix-free flavors this comp
    can run under OpenMDAO's Newton solver, compute totals in both directions,
    and undergo a full `check_partials` (which requires both directions for
    matrix-free components).
    """

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_matrix_free_implicit_fwd_rev_test_comp(n=n, a=a)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=False)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_matrix_free_fwd_rev_ad", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_partials_both_directions(self):
        # `check_partials` on matrix-free components checks both the "fwd" and
        # "rev" directions against the finite-difference approximation; this
        # only works when the comp supports both directions.
        _check_partials_vs_analytic(self.p, "icomp_matrix_free_fwd_rev_ad", self.x, self.y, self.a)

    def test_totals(self):
        p = self.p
        totals = p.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
        expected = _expected_totals(self.x, self.y, self.a)
        for (of, wrt), J_analytic in expected.items():
            np.testing.assert_allclose(totals[of, wrt], J_analytic, rtol=1e-6, atol=1e-8)

    def test_apply_linear_rev(self):
        comp = self.p.model.icomp_matrix_free_fwd_rev_ad
        n = self.n
        a = self.a
        x = self.x
        y = self.y

        inputs = {"x": np.array([x]), "y": y.copy()}
        outputs = {"z1": np.zeros(n), "z2": np.zeros(1)}
        d_inputs = {"x": np.zeros(1), "y": np.zeros(n)}
        d_outputs = {"z1": np.zeros(n), "z2": np.zeros(1)}
        d_residuals = {"z1": np.ones(n), "z2": np.zeros(1)}

        comp.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "rev")

        # Same VJP expectations as the reverse-only test: with seed
        # dR.z1 = ones(n), dx = n*2*a*x and dy = 2*y (diagonal).
        np.testing.assert_allclose(d_inputs["x"], np.array([n * 2.0 * a * x]), rtol=1e-10)
        np.testing.assert_allclose(d_inputs["y"], 2.0 * y, rtol=1e-10)
        np.testing.assert_allclose(d_outputs["z1"], -np.ones(n), rtol=1e-10)
        np.testing.assert_allclose(d_outputs["z2"], np.zeros(1), rtol=1e-10)


@use_tempdirs
class TestDenseImplicitADUnitsTags(unittest.TestCase):
    """Verify that `units_dict`/`tags_dict` metadata propagates from the Julia
    component through omjlcomps into OpenMDAO's variable metadata.

    `y` and `z2` are deliberately left out of `units_dict`/`tags_dict` so the
    defaults ("unitless" units and no tags) are covered as well.
    """

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0

        jlcomp = jl.get_dense_implicit_units_tags_test_comp(n=n, a=a)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("icomp_units_tags", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup()

    def test_units_and_tags(self):
        comp = self.p.model.icomp_units_tags

        expected_units = {"x": "m", "y": "unitless", "z1": "kg", "z2": "unitless"}
        expected_tags = {
            "x": {"tag_x"},
            "y": set(),
            "z1": {"tag_z1a", "tag_z1b"},
            "z2": set(),
        }

        meta = comp._var_allprocs_abs2meta
        for iotype in ("input", "output"):
            for name, m in meta[iotype].items():
                var = name.split(".")[-1]
                self.assertEqual(m["units"], expected_units[var],
                                 msg=f"wrong units for {iotype} var {var}")
                # Filter out tags that OpenMDAO itself adds (e.g. the
                # "openmdao:allow_desvar" tag on outputs).
                user_tags = {t for t in m["tags"] if not t.startswith("openmdao:")}
                self.assertEqual(user_tags, expected_tags[var],
                                 msg=f"wrong tags for {iotype} var {var}")


@use_tempdirs
class TestDenseImplicitADShapeByConn(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        scale = self.scale = 2.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_shape_by_conn_test_comp(n=n, a=a, scale=scale)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2,
                                                err_on_non_converge=True)
        p.model.add_subsystem("icomp_dense_ad_shape_by_conn", comp, promotes_inputs=["y"],
                              promotes_outputs=["z"])
        p.setup(force_alloc_complex=True)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z_expected = self.scale * self.y**2 + self.a
        np.testing.assert_allclose(p.get_val("z"), z_expected, rtol=1e-10)

    def test_partials_fd(self):
        p = self.p
        n = self.n
        a = self.a
        scale = self.scale
        y = self.y

        cpd = p.check_partials(compact_print=True, out_stream=None, method="fd")
        comp_partials = cpd["icomp_dense_ad_shape_by_conn"]

        expected = {
            ("z", "y"): np.diag(2.0 * scale * y),
            ("z", "z"): -np.eye(n),
        }
        for (of, wrt), J_analytic in expected.items():
            J = comp_partials[of, wrt]["J_fd"]
            np.testing.assert_allclose(J, J_analytic, rtol=1e-6, atol=1e-8)


@use_tempdirs
class TestDenseImplicitADGuessInPlace(unittest.TestCase):
    """End-to-end test of `guess_nonlinear` for the in-place dense AD implicit comp."""

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_guess_test_comp(n=n, a=a, in_place=True)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=0,
                                                err_on_non_converge=True, maxiter=10)
        p.model.add_subsystem("icomp_dense_ad_guess", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_newton_converges_immediately(self):
        # The guess provides the exact solution, so Newton should converge on
        # its first residual check. Compare against the same comp *without* a
        # guess function, which needs a full linear-solve iteration.
        p = self.p
        iter_count_guess = p.model.icomp_dense_ad_guess.nonlinear_solver._iter_count

        p_noguess = om.Problem()
        jlcomp = jl.get_dense_implicit_test_comp(n=self.n, a=self.a, in_place=True)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=0,
                                                err_on_non_converge=True, maxiter=10)
        p_noguess.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"],
                                      promotes_outputs=["z1", "z2"])
        p_noguess.setup()
        p_noguess.set_val("x", self.x)
        p_noguess.set_val("y", self.y)
        p_noguess.run_model()
        iter_count_noguess = p_noguess.model.icomp.nonlinear_solver._iter_count

        self.assertLess(iter_count_guess, iter_count_noguess)

    def test_direct_guess_call(self):
        # Call `guess_nonlinear` directly, with a scalar `z2` entry, to
        # exercise the copy-back of scalar outputs from Julia. Note that the
        # residuals are read-only in `guess_nonlinear` (OpenMDAO contract), so
        # they are not modified even though the guess function zeroes them.
        comp = self.p.model.icomp_dense_ad_guess
        inputs = {"x": np.array(self.x), "y": self.y.copy()}
        outputs = {"z1": np.zeros(self.n), "z2": 0.0}
        residuals = {"z1": np.full(self.n, 99.0), "z2": 99.0}
        comp.guess_nonlinear(inputs, outputs, residuals)

        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(outputs["z1"], z1_expected, rtol=1e-10)
        np.testing.assert_allclose(outputs["z2"], z2_expected, rtol=1e-10)
        np.testing.assert_allclose(residuals["z1"], 99.0)  # untouched
        np.testing.assert_allclose(residuals["z2"], 99.0)  # read-only


@use_tempdirs
class TestDenseImplicitADGuessOutOfPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_guess_test_comp(n=n, a=a, in_place=False)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=0,
                                                err_on_non_converge=True, maxiter=10)
        p.model.add_subsystem("icomp_dense_ad_guess", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)


@use_tempdirs
class TestDenseImplicitADGuessScalarResidual(unittest.TestCase):
    """Guess function that only updates the scalar state `z2`, to verify that
    the scalar output copy-back in omjlcomps' `guess_nonlinear` override
    works and that untouched entries pass through unchanged."""

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 5
        a = self.a = 3.0
        self.x = 3.0
        self.y = np.arange(1.0, n + 1.0)

        jlcomp = jl.get_dense_implicit_guess_test_comp(n=n, a=a, in_place=True, scalar_only=True)
        comp = JuliaImplicitComp(jlcomp=jlcomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=0,
                                                err_on_non_converge=True, maxiter=10)
        p.model.add_subsystem("icomp_dense_ad_guess_scalar", comp, promotes_inputs=["x", "y"],
                              promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", self.x)
        p.set_val("y", self.y)
        p.run_model()

    def test_results(self):
        p = self.p
        z1_expected, z2_expected = _expected_states(self.x, self.y, self.a)
        np.testing.assert_allclose(p.get_val("z1"), z1_expected, rtol=1e-10)
        np.testing.assert_allclose(p.get_val("z2"), z2_expected, rtol=1e-10)

    def test_direct_guess_call(self):
        comp = self.p.model.icomp_dense_ad_guess_scalar
        inputs = {"x": np.array(self.x), "y": self.y.copy()}
        outputs = {"z1": np.zeros(self.n), "z2": 0.0}
        residuals = {"z1": np.full(self.n, 99.0), "z2": 99.0}
        comp.guess_nonlinear(inputs, outputs, residuals)

        # Only the scalar state z2 was touched by the guess function.
        np.testing.assert_allclose(outputs["z1"], 0.0)  # untouched
        np.testing.assert_allclose(outputs["z2"], self.a * self.x + np.sum(self.y), rtol=1e-10)
        np.testing.assert_allclose(residuals["z1"], 99.0)  # untouched
        np.testing.assert_allclose(residuals["z2"], 99.0)  # read-only


if __name__ == "__main__":
    unittest.main()
