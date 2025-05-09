"""
Unit tests for JuliaExplicitComp
"""
import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaExplicitCompTest")
import os
import sys
import time
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.core.analysis_error import AnalysisError


from omjlcomps import JuliaExplicitComp

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_ecomp.jl"))


class TestSimpleJuliaExplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.ECompTest.ECompSimple()
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

class TestSimpleJuliaExplicitCompWithRecording(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.ECompTest.ECompSimple()
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.add_recorder(om.SqliteRecorder("simple_explicit.sql"))
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()
        p.record("final")
        p.cleanup()


    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

        d = p.get_reports_dir()
        cr = om.CaseReader(f"{d}/../simple_explicit.sql")
        c = cr.get_case("final")
        actual_rec = np.squeeze(c.get_val("y"))
        np.testing.assert_almost_equal(actual_rec, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

class TestSimpleJuliaExplicitCSComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.ECompTest.ECompSimpleCS()
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='fd')

        # Check that the partials calculated by OpenMDAO's complex step match the exact.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the finite difference check.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=5)

class TestJuliaExplicitCompWithOption(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        a = self.a = 0.5
        ecomp = jl.ECompTest.ECompWithOption(a)
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*self.a*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*self.a*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

class TestJuliaExplicitCompWithOptionAndTags(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        a = self.a = 0.5
        ecomp = jl.ECompTest.ECompWithOptionAndTags(a, xtag=juliacall.convert(jl.Vector, ["xtag1", "xtag2"]), ytag=juliacall.convert(jl.Vector, ["ytag"]))
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results1(self):
        p = self.p
        expected = 2*self.a*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials1(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*self.a*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

class TestJuliaExplicitCompWithGlobs(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        a = self.a = 0.5
        ecomp = jl.ECompTest.ECompWithGlobs(a, xtag=juliacall.convert(jl.Vector, ["xtag1", "xtag2"]), ytag=juliacall.convert(jl.Vector, ["ytag"]))
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results1(self):
        p = self.p
        expected = 2*self.a*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials1(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*self.a*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

class TestJuliaExplicitCompWithLargeOption(unittest.TestCase):

    def setUp(self):
        n_small = self.n_small = 10
        p_small = self.p_small = om.Problem()
        ecomp_small = jl.ECompTest.ECompWithLargeOption(n_small)
        comp_small = JuliaExplicitComp(jlcomp=ecomp_small)
        p_small.model.add_subsystem("ecomp", comp_small, promotes_inputs=["x"], promotes_outputs=["y"])
        p_small.setup(force_alloc_complex=True)
        p_small.set_val("x", 3.0)
        p_small.run_model()

        if os.getenv("GITHUB_ACTIONS") == "true":
            # n_big = 1_000_000_000 is too big for GitHub Actions---it appears to consume all the virtual machine's memory and kills the job, giving me a big sad.
            if sys.platform == "darwin":
                n_big = self.n_big = 20_000_000
            else:
                n_big = self.n_big = 100_000_000
        else:
            n_big = self.n_big = 500_000_000
        p_big = self.p_big = om.Problem()
        ecomp_big = jl.ECompTest.ECompWithLargeOption(n_big)
        comp_big = JuliaExplicitComp(jlcomp=ecomp_big)
        p_big.model.add_subsystem("ecomp", comp_big, promotes_inputs=["x"], promotes_outputs=["y"])
        p_big.setup(force_alloc_complex=True)
        p_big.set_val("x", 3.0)
        p_big.run_model()

    def test_results(self):
        for p in [self.p_small, self.p_big]:
            expected = 2*3*p.get_val("x")[0]**2 + 1
            actual = p.get_val("y")[0]
            np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        for p in [self.p_small, self.p_big]:
            np.set_printoptions(linewidth=1024)
            cpd = p.check_partials(compact_print=True, out_stream=None, method='cs')

            # Check that the partials the user provided are correct.
            ecomp_partials = cpd["ecomp"]
            np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*3*p.get_val("x")[0]]], decimal=12)

            # Check that partials approximated by the complex-step method match the user-provided partials.
            for comp in cpd:
                for (var, wrt) in cpd[comp]:
                    np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                                   desired=cpd[comp][var, wrt]['J_fd'],
                                                   decimal=12)

    def test_timings(self):
        time_avg = []
        n_samples = 1000
        for p in [self.p_small, self.p_big]:
            # Just to make sure the Julia JIT is all warmed up.
            p.run_model()
            tavg = 0.0
            for sample in range(n_samples):
                tstart = time.time()
                p.run_model()
                telapsed = time.time() - tstart
                tavg += telapsed/n_samples

            time_avg.append(tavg)

        # Compare the average timings.
        np.testing.assert_almost_equal(time_avg[1]/time_avg[0], 1.0, decimal=1)


class TestJuliaMatrixFreeExplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        nrows, ncols = 2, 3
        ecomp = jl.ECompTest.ECompMatrixFree(nrows, ncols)
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x1", "x2"], promotes_outputs=["y1", "y2"])
        p.setup(force_alloc_complex=True)
        p.set_val("x1", np.arange(nrows*ncols).reshape(nrows, ncols)+0.5)
        p.set_val("x2", np.arange(nrows*ncols).reshape(nrows, ncols)+1)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x1") + 3*p.get_val("x2")**2
        actual = p.get_val("y1")
        np.testing.assert_almost_equal(actual, expected)

        expected = 4*p.get_val("x1")**3 + 5*p.get_val("x2")**4
        actual = p.get_val("y2")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_allclose(actual=cpd[comp][var, wrt]['J_fwd'], desired=cpd[comp][var, wrt]['J_fd'], rtol=1e-12)
                np.testing.assert_allclose(actual=cpd[comp][var, wrt]['J_rev'], desired=cpd[comp][var, wrt]['J_fd'], rtol=1e-12)

        p.set_val("x1", np.arange(2*3).reshape(2,3)+4)
        p.set_val("x2", np.arange(2*3).reshape(2,3)+5)

        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_allclose(actual=cpd[comp][var, wrt]['J_fwd'], desired=cpd[comp][var, wrt]['J_fd'], rtol=1e-12)
                np.testing.assert_allclose(actual=cpd[comp][var, wrt]['J_rev'], desired=cpd[comp][var, wrt]['J_fd'], rtol=1e-12)


class TestShapeByConn(unittest.TestCase):
    def setUp(self):
        p = self.p = om.Problem()
        n = 8
        comp = om.IndepVarComp()
        comp.add_output("x", shape=(n, n))
        p.model.add_subsystem("inputs_comp", comp, promotes_outputs=["x"])

        comp = JuliaExplicitComp(jlcomp=jl.ECompTest.ECompShapeByConn())
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])

        p.setup(force_alloc_complex=True)
        p.set_val("x", np.arange(n))
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")**2 + 1
        actual = p.get_val("y")
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_allclose(actual=cpd[comp][var, wrt]['J_fwd'], desired=cpd[comp][var, wrt]['J_fd'], rtol=1e-12)


class TestAnalysisError(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.ECompTest.ECompDomainError()
        comp = JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        p.set_val("x", 3.0)
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        p.set_val("x", 3.0)
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        ecomp_partials = cpd["ecomp"]
        np.testing.assert_almost_equal(actual=ecomp_partials["y", "x"]['J_fwd'], desired=[[4*p.get_val("x")[0]]], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)

    def test_analysis_error(self):
        p = self.p
        p.set_val("x", -1.0)
        self.assertRaises(AnalysisError, p.run_model)
        self.assertRaises(AnalysisError, p.check_partials)


if __name__ == '__main__':
    unittest.main()
