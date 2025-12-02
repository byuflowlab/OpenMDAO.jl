import juliacall

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from omjlcomps import JuliaExplicitComp
from example_python_package_openmdao_jl.paraboloid import get_parabaloid_comp


class TestParaboloidExample(unittest.TestCase):
    def setUp(self):
        prob = self.p = om.Problem()
        jlcomp = get_parabaloid_comp()
        parab_comp = JuliaExplicitComp(jlcomp=jlcomp)

        prob.model.add_subsystem("parab_comp", parab_comp)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"

        prob.model.add_design_var("parab_comp.x")
        prob.model.add_design_var("parab_comp.y")
        prob.model.add_objective("parab_comp.f_xy")

        prob.setup(force_alloc_complex=True)

    def test_run_model(self):
        prob = self.p
        prob.set_val("parab_comp.x", 3.0)
        prob.set_val("parab_comp.y", -4.0)
        prob.run_model()
        f = prob.get_val("parab_comp.f_xy")
        np.testing.assert_almost_equal(f, -15.0)

        prob.set_val("parab_comp.x", 5.0)
        prob.set_val("parab_comp.y", -2.0)
        prob.run_model()
        f = prob.get_val("parab_comp.f_xy")
        np.testing.assert_almost_equal(f, -5.0)

    def test_partials(self):
        prob = self.p
        prob.set_val("parab_comp.x", 3.0)
        prob.set_val("parab_comp.y", -4.0)
        data = prob.check_partials(method="cs")
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_run_driver(self):
        prob = self.p
        prob.set_val("parab_comp.x", 5.0)
        prob.set_val("parab_comp.y", -2.0)
        prob.run_model()
        prob.run_driver()
        f = prob.get_val("parab_comp.f_xy")
        np.testing.assert_almost_equal(f, -27.333333333)
        x = prob.get_val("parab_comp.x")
        np.testing.assert_almost_equal(x, 6.666666666666666)
        y = prob.get_val("parab_comp.y")
        np.testing.assert_almost_equal(y, -7.333333333333)


if __name__ == '__main__':
    unittest.main()
