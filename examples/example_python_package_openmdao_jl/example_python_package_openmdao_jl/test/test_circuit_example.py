import juliacall

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from omjlcomps import JuliaExplicitComp
from example_python_package_openmdao_jl.circuit import get_circuit_comp


class TestCircuitExample(unittest.TestCase):

    def setUp(self):
        prob = self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output("I_in", val=0.1, shape=(), units="A")
        ivc.add_output("V_g", val=0.0, shape=(), units="V")
        prob.model.add_subsystem("ivc", ivc, promotes=["*"])

        Is=1e-15
        Vt=0.025875
        jlcomp = get_circuit_comp(Is, Vt)
        circuit_comp = JuliaExplicitComp(jlcomp=jlcomp)
        prob.model.add_subsystem(
            "circuit_comp", circuit_comp,
            promotes_inputs=["I_in", "V_g", "R1", "R2"],
            promotes_outputs=["n1_V", "n2_V"])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"

        # prob.model.add_design_var("circuit_comp.I_in")
        prob.model.add_design_var("R1")
        prob.model.add_design_var("R2")
        # prob.model.add_design_var("circuit_comp.V_g")
        prob.model.add_objective("n1_V")
        prob.model.add_constraint("n2_V", equals=0.5, units="V")

        prob.setup(force_alloc_complex=False)

    def test_run_model(self):
        prob = self.p
        prob.set_val("I_in", 0.1, units="A")
        prob.set_val("V_g", 0.0, units="V")
        prob.set_val("R1", 100.0, units="ohm")
        prob.set_val("R2", 10000.0, units="ohm")
        prob.run_model()
        n1_V = prob.get_val("n1_V", units="V")
        n2_V = prob.get_val("n2_V", units="V")
        n1_V_expected = 9.90804735
        n2_V_expected = 0.71278185
        np.testing.assert_almost_equal(n1_V, n1_V_expected)
        np.testing.assert_almost_equal(n2_V, n2_V_expected)

    def test_partials(self):
        prob = self.p
        prob.set_val("I_in", 0.1, units="A")
        prob.set_val("V_g", 0.0, units="V")
        prob.set_val("R1", 100.0, units="ohm")
        prob.set_val("R2", 10000.0, units="ohm")
        data = prob.check_partials()
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_run_driver(self):
        prob = self.p
        prob.set_val("I_in", 0.1, units="A")
        prob.set_val("V_g", 0.0, units="V")
        prob.set_val("R1", 100.0, units="ohm")
        prob.set_val("R2", 10000.0, units="ohm")
        prob.run_model()
        prob.run_driver()
        I_in = prob.get_val("I_in", units="A")
        V_g = prob.get_val("V_g", units="V")
        R1 = prob.get_val("R1", units="ohm")
        R2 = prob.get_val("R2", units="ohm")
        n1_V = prob.get_val("n1_V", units="V")
        n2_V = prob.get_val("n2_V", units="V")
        print(f"I_in = {I_in} A (fixed)")
        print(f"V_g = {V_g} V (fixed)")
        print(f"R1 = {R1} Ohms (design variable)")
        print(f"R2 = {R2} Ohms (design variable)")
        print(f"n1_V = {n1_V} V (objective)")
        print(f"n2_V = {n2_V} V (constrainted to 0.5 V)")

        R1_expected = 5.024564
        R2_expected = 10000.72332092
        n1_V_expected = 0.5024551666550083
        n2_V_expected = 0.49999987756278375
        np.testing.assert_almost_equal(R1, R1_expected)
        np.testing.assert_almost_equal(R2, R2_expected)
        np.testing.assert_almost_equal(n1_V, n1_V_expected)
        np.testing.assert_almost_equal(n2_V, n2_V_expected)


if __name__ == '__main__':
    unittest.main()
