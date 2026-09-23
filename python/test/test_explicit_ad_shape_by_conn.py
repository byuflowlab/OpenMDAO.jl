import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaADExplictShapeByConnTests")
import os
import unittest

import numpy as np
from numpy.random import rand

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.core.analysis_error import AnalysisError

from omjlcomps import JuliaExplicitComp


d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_explicit_ad_shape_by_conn.jl"))


class TestExplicitADSparseInPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()

        x_shape = self._x_shape = (2, 3, 4)
        comp = om.IndepVarComp("x", shape=x_shape)
        p.model.add_subsystem("ivc", comp, promotes=["*"])

        jlcomp = jl.get_sparse_test_comp(in_place=True)
        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=["x"], promotes_outputs=["y", "z"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(np.prod(x_shape)).reshape(x_shape))
        p.run_model()

    def test_results(self):
        p = self.p
        x_actual = p.get_val("x")
        y_expected = 0.5*x_actual
        y_actual = p.get_val("y")
        np.testing.assert_almost_equal(y_actual, y_expected)

        x_actual_rev = x_actual.flatten()[::-1].reshape(self._x_shape)
        z_expected = 3.0 * x_actual_rev*x_actual**2 + 4*np.sum(x_actual, -1, keepdims=True) + 2*x_actual[0, 1, 0]*x_actual[1, 0, 1]
        z_actual = p.get_val("z")
        np.testing.assert_almost_equal(z_actual, z_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                J_fd = cpd[comp][var, wrt]['J_fd']
                J_fwd = cpd[comp][var, wrt]['J_fwd']
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestMatrixFreeADInPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()

        x_shape = self._x_shape = (2, 3, 4)
        comp = om.IndepVarComp("x", shape=x_shape)
        p.model.add_subsystem("ivc", comp, promotes=["*"])

        jlcomp = jl.get_matrix_free_test_comp(in_place=True)
        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=["x"], promotes_outputs=["y", "z"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(np.prod(x_shape)).reshape(x_shape))
        p.run_model()

    def test_results(self):
        p = self.p
        x_actual = p.get_val("x")
        y_expected = 0.5*x_actual
        y_actual = p.get_val("y")
        np.testing.assert_almost_equal(y_actual, y_expected)

        x_actual_rev = x_actual.flatten()[::-1].reshape(self._x_shape)
        z_expected = 3.0 * x_actual_rev*x_actual**2 + 4*np.sum(x_actual, -1, keepdims=True) + 2*x_actual[0, 1, 0]*x_actual[1, 0, 1]
        z_actual = p.get_val("z")
        np.testing.assert_almost_equal(z_actual, z_expected)


    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                J_fd = cpd[comp][var, wrt]['J_fd']
                J_fwd = cpd[comp][var, wrt]['J_fwd']
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestExplicitADSparseOutOfPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()

        x_shape = self._x_shape = (2, 3, 4)
        comp = om.IndepVarComp("x", shape=x_shape)
        p.model.add_subsystem("ivc", comp, promotes=["*"])

        jlcomp = jl.get_sparse_test_comp(in_place=False)
        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=["x"], promotes_outputs=["y", "z"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(np.prod(x_shape)).reshape(x_shape))
        p.run_model()

    def test_results(self):
        p = self.p
        x_actual = p.get_val("x")
        y_expected = 0.5*x_actual
        y_actual = p.get_val("y")
        np.testing.assert_almost_equal(y_actual, y_expected)

        x_actual_rev = x_actual.flatten()[::-1].reshape(self._x_shape)
        z_expected = 3.0 * x_actual_rev*x_actual**2 + 4*np.sum(x_actual, -1, keepdims=True) + 2*x_actual[0, 1, 0]*x_actual[1, 0, 1]
        z_actual = p.get_val("z")
        np.testing.assert_almost_equal(z_actual, z_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                J_fd = cpd[comp][var, wrt]['J_fd']
                J_fwd = cpd[comp][var, wrt]['J_fwd']
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestMatrixFreeADOutOfPlace(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()

        x_shape = self._x_shape = (2, 3, 4)
        comp = om.IndepVarComp("x", shape=x_shape)
        p.model.add_subsystem("ivc", comp, promotes=["*"])

        jlcomp = jl.get_matrix_free_test_comp(in_place=False)
        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=["x"], promotes_outputs=["y", "z"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(np.prod(x_shape)).reshape(x_shape))
        p.run_model()

    def test_results(self):
        p = self.p
        x_actual = p.get_val("x")
        y_expected = 0.5*x_actual
        y_actual = p.get_val("y")
        np.testing.assert_almost_equal(y_actual, y_expected)

        x_actual_rev = x_actual.flatten()[::-1].reshape(self._x_shape)
        z_expected = 3.0 * x_actual_rev*x_actual**2 + 4*np.sum(x_actual, -1, keepdims=True) + 2*x_actual[0, 1, 0]*x_actual[1, 0, 1]
        z_actual = p.get_val("z")
        np.testing.assert_almost_equal(z_actual, z_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                J_fd = cpd[comp][var, wrt]['J_fd']
                J_fwd = cpd[comp][var, wrt]['J_fwd']
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


if __name__ == '__main__':
    unittest.main()
