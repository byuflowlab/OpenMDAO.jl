"""
Unit tests for JuliaImplicitComp
"""
import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaImplicitCompTest")
import os
import time
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from omjlcomps import JuliaImplicitComp

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_icomp.jl"))


class TestSimpleJuliaImplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 10
        a = self.a = 3.0
        icomp = jl.ICompTest.SimpleImplicit(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

    def test_results(self):
        p = self.p
        a = self.a
        expected = a*p.get_val("x")**2 + p.get_val("y")**2
        actual = p.get_val("z1")
        np.testing.assert_almost_equal(actual, expected)
        expected = a*p.get_val("x") + p.get_val("y")
        actual = p.get_val("z2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        icomp_partials = cpd["icomp"]

        actual = icomp_partials["z1", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*p.get_val("y")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "z1"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = self.a
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "z2"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestSimpleJuliaImplicitWithGlobComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 10
        a = self.a = 3.0
        icomp = jl.ICompTest.SimpleImplicitWithGlob(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

    def test_results(self):
        p = self.p
        a = self.a
        expected = a*p.get_val("x")**2 + p.get_val("y")**2
        actual = p.get_val("z1")
        np.testing.assert_almost_equal(actual, expected)
        expected = a*p.get_val("x") + p.get_val("y")
        actual = p.get_val("z2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        icomp_partials = cpd["icomp"]

        actual = icomp_partials["z1", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*p.get_val("y")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "z1"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = self.a
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "z2"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)



class TestSolveNonlinearJuliaImplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 10
        a = self.a = 3.0
        icomp = jl.ICompTest.SolveNonlinearImplicit(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

    def test_results(self):
        p = self.p
        a = self.a
        expected = a*p.get_val("x")**2 + p.get_val("y")**2
        actual = p.get_val("z1")
        np.testing.assert_almost_equal(actual, expected)
        expected = a*p.get_val("x") + p.get_val("y")
        actual = p.get_val("z2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        icomp_partials = cpd["icomp"]

        actual = icomp_partials["z1", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*p.get_val("y")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "z1"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = self.a
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "z2"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestMatrixFreeImplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 10
        a = self.a = 3.0
        icomp = jl.ICompTest.MatrixFreeImplicit(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=False)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(n)+0.5)
        p.set_val("y", np.arange(n)+2)
        p.run_model()

    def test_results(self):
        p = self.p
        a = self.a
        expected = a*p.get_val("x")**2 + p.get_val("y")**2
        actual = p.get_val("z1")
        np.testing.assert_almost_equal(actual, expected)
        expected = a*p.get_val("x") + p.get_val("y")
        actual = p.get_val("z2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        icomp_partials = cpd["icomp"]

        actual = icomp_partials["z1", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*p.get_val("y")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "z1"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = self.a
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "z2"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_rev'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestSolveLinearImplicitComp(unittest.TestCase):

    def setUp(self):
        n = self.n = 10
        a = self.a = 3.0

        p = self.p_fwd = om.Problem()
        icomp = jl.ICompTest.SolveLinearImplicit(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])

        p.setup(force_alloc_complex=True, mode="fwd")
        p.set_val("x", np.arange(n)+0.5)
        p.set_val("y", np.arange(n)+2)
        p.run_model()

        p = self.p_rev = om.Problem()
        icomp = jl.ICompTest.SolveLinearImplicit(n, a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])

        p.setup(force_alloc_complex=True, mode="rev")
        p.set_val("x", np.arange(n)+0.5)
        p.set_val("y", np.arange(n)+2)
        p.run_model()

    def test_results(self):
        for p in [self.p_fwd, self.p_rev]:
            a = self.a
            expected = a*p.get_val("x")**2 + p.get_val("y")**2
            actual = p.get_val("z1")
            np.testing.assert_almost_equal(actual, expected)
            expected = a*p.get_val("x") + p.get_val("y")
            actual = p.get_val("z2")
            np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        for p in [self.p_fwd, self.p_rev]:
            np.set_printoptions(linewidth=1024)
            cpd = p.check_partials(compact_print=True, out_stream=None, method='cs')

            # Check that the partials the user provided are correct.
            icomp_partials = cpd["icomp"]

            actual = icomp_partials["z1", "x"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            actual = icomp_partials["z1", "y"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = 2*p.get_val("y")
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            actual = icomp_partials["z1", "z1"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = -1.0
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            actual = icomp_partials["z2", "x"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = self.a
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            actual = icomp_partials["z2", "y"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = 1.0
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            actual = icomp_partials["z2", "z2"]['J_fwd']
            expected = np.zeros((self.n, self.n))
            expected[range(self.n), range(self.n)] = -1.0
            np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

            # Check that partials approximated by the complex-step method match the user-provided partials.
            for comp in cpd:
                for (var, wrt) in cpd[comp]:
                    np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                                   desired=cpd[comp][var, wrt]['J_fd'],
                                                   decimal=12)
                    # np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_rev'],
                    #                                desired=cpd[comp][var, wrt]['J_fd'],
                    #                                decimal=12)

    def test_totals(self):
        for p in [self.p_fwd, self.p_rev]:
            ctd = p.check_totals(of=["z1", "z2"], wrt=["x", "y"], method='cs', compact_print=True, out_stream=None)
            for of_wrt in ctd:
                np.testing.assert_almost_equal(actual=ctd[of_wrt]['J_fwd'],
                                               desired=ctd[of_wrt]['J_fd'],
                                               decimal=12)



class TestGuessNonlinearImplicitComp(unittest.TestCase):

    def setUp(self):
        n = self.n = 1

        # Create a component that should find the left root of R(x) = x**2 - 4*x + 3 = (x - 1)*(x - 3) = 0, aka 1.
        # (x - 1)*(x - 3)
        xguess = 1.5
        xlower = -10.0
        xupper = 1.9
        p_leftroot = self.p_leftroot = om.Problem()
        icomp = jl.ICompTest.GuessNonlinearImplicit(n, xguess, xlower, xupper)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        comp.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar')
        p_leftroot.model.add_subsystem("icomp", comp, promotes_inputs=["a", "b", "c"], promotes_outputs=["x"])
        p_leftroot.setup(force_alloc_complex=True)
        p_leftroot.set_val("a", 1.0)
        p_leftroot.set_val("b", -4.0)
        p_leftroot.set_val("c", 3.0)
        p_leftroot.run_model()

        # Create a component that should find the right root of R(x) = x**2 - 4*x + 3 = 0, aka 3.
        xguess = 2.5
        xlower = 2.1
        xupper = 10.0
        p_rightroot = self.p_rightroot = om.Problem()
        icomp = jl.ICompTest.GuessNonlinearImplicit(n, xguess, xlower, xupper)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        comp.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar')
        p_rightroot.model.add_subsystem("icomp", comp, promotes_inputs=["a", "b", "c"], promotes_outputs=["x"])
        p_rightroot.setup(force_alloc_complex=True)
        p_rightroot.set_val("a", 1.0)
        p_rightroot.set_val("b", -4.0)
        p_rightroot.set_val("c", 3.0)
        p_rightroot.run_model()

    def test_results(self):
        expected = 1*np.ones(self.n)
        actual = self.p_leftroot.get_val("x")
        np.testing.assert_almost_equal(actual, expected)

        expected = 3*np.ones(self.n)
        actual = self.p_rightroot.get_val("x")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        for p in [self.p_leftroot, self.p_rightroot]:
            cpd = p.check_partials(compact_print=True, out_stream=None, method='cs')

            # Check that partials approximated by the complex-step method match the user-provided partials.
            for comp in cpd:
                for (var, wrt) in cpd[comp]:
                    np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                                   desired=cpd[comp][var, wrt]['J_fd'],
                                                   decimal=12)


class TestShapeByConn(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        n = self.n = 10
        a = self.a = 3.0
        comp = om.IndepVarComp()
        comp.add_output("x", shape=n)
        comp.add_output("y", shape=n)
        p.model.add_subsystem("input_comp", comp, promotes_outputs=["x", "y"])
        icomp = jl.ICompTest.ImplicitShapeByConn(a)
        comp = JuliaImplicitComp(jlcomp=icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

    def test_results(self):
        p = self.p
        a = self.a
        expected = a*p.get_val("x")**2 + p.get_val("y")**2
        actual = p.get_val("z1")
        np.testing.assert_almost_equal(actual, expected)
        expected = a*p.get_val("x") + p.get_val("y")
        actual = p.get_val("z2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        icomp_partials = cpd["icomp"]

        actual = icomp_partials["z1", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*self.a*p.get_val("x")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 2*p.get_val("y")
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z1", "z1"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "x"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = self.a
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "y"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = 1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        actual = icomp_partials["z2", "z2"]['J_fwd']
        expected = np.zeros((self.n, self.n))
        expected[range(self.n), range(self.n)] = -1.0
        np.testing.assert_almost_equal(actual=actual, desired=expected, decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)



if __name__ == '__main__':
    unittest.main()
