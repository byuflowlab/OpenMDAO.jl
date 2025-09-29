import juliacall

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from omjlcomps import JuliaExplicitComp

from example_python_package_openmdao_jl.circle import (get_arctan_yox_comp,
                                                       get_circle_comp,
                                                       get_r_con_comp,
                                                       get_theta_con_comp,
                                                       get_delta_theta_con_comp,
                                                       get_l_conx_comp)

class TestCircleExample(unittest.TestCase):
    def setUp(self):
        # Use the same value of `SIZE` that the official version uses:
        SIZE = 10

        self.p = p = om.Problem()

        jlcomp = get_arctan_yox_comp(SIZE)
        arctan_yox_comp = JuliaExplicitComp(jlcomp=jlcomp)

        jlcomp = get_circle_comp()
        circle_comp = JuliaExplicitComp(jlcomp=jlcomp)

        jlcomp = get_r_con_comp(SIZE)
        r_con_comp = JuliaExplicitComp(jlcomp=jlcomp)

        jlcomp = get_theta_con_comp(SIZE)
        theta_con_comp = JuliaExplicitComp(jlcomp=jlcomp)

        jlcomp = get_delta_theta_con_comp(SIZE)
        delta_theta_con_comp = JuliaExplicitComp(jlcomp=jlcomp)

        jlcomp = get_l_conx_comp(SIZE)
        l_conx_comp = JuliaExplicitComp(jlcomp=jlcomp)

        p.model.add_subsystem("arctan_yox", arctan_yox_comp, promotes_inputs=["x", "y"])
        p.model.add_subsystem("circle", circle_comp, promotes_inputs=["r"])
        p.model.add_subsystem("r_con", r_con_comp, promotes_inputs=["r", "x", "y"])
        p.model.add_subsystem("theta_con", theta_con_comp)
        p.model.add_subsystem("delta_theta_con", delta_theta_con_comp)
        p.model.add_subsystem("l_conx", l_conx_comp, promotes_inputs=["x"])

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        p.model.connect("arctan_yox.g", "theta_con.x")
        p.model.connect("arctan_yox.g", "delta_theta_con.even", src_indices=EVEN_IND)
        p.model.connect("arctan_yox.g", "delta_theta_con.odd", src_indices=ODD_IND)

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options["optimizer"] = "SLSQP"
        p.driver.options["disp"] = False

        # set up dynamic total coloring here
        p.driver.declare_coloring()

        p.model.add_design_var("x")
        p.model.add_design_var("y")
        p.model.add_design_var("r", lower=.5, upper=10)

        # nonlinear constraints
        p.model.add_constraint("r_con.g", equals=0)

        p.model.add_constraint("theta_con.g", lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        p.model.add_constraint("delta_theta_con.g", lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        p.model.add_constraint("l_conx.g", equals=0, linear=False, indices=[0,])

        # linear constraint
        p.model.add_constraint("y", equals=0, indices=[0,], linear=True)

        p.model.add_objective("circle.area", ref=-1)

        p.setup(force_alloc_complex=True)

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        p.set_val('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                  0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        p.set_val('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                 -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        p.set_val('r', .7)

    def test_partials(self):
        prob = self.p
        # prob.set_val("parab_comp.x", 3.0)
        # prob.set_val("parab_comp.y", -4.0)
        data = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_run_driver(self):
        prob = self.p
        prob.run_driver()
        np.testing.assert_almost_equal(prob["circle.area"], np.pi)



if __name__ == '__main__':
    unittest.main()
