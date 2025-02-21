import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaAviarySupportTests")
import os
import unittest

import aviary.api as av

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.core.analysis_error import AnalysisError

from omjlcomps import JuliaExplicitComp, to_jlsymstrdict


d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_aviary_support.jl"))


class TestAviaryMatrixFreeInPlacePropellerRadiusComp(unittest.TestCase):

    def setUp(self):
        av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["default_value"] = 2.3

        p = self.p = om.Problem()
        jlcomp = jl.get_aviary_matrix_free_test_comp(
            in_place=True,
            aviary_input_names=to_jlsymstrdict({"Dtip": av.Aircraft.Engine.Propeller.DIAMETER}),
            aviary_output_names=to_jlsymstrdict({"thrust": av.Dynamic.Vehicle.Propulsion.THRUST}),
            aviary_meta_data=av.CoreMetaData)

        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=[av.Aircraft.Engine.Propeller.DIAMETER], promotes_outputs=["Rtip", av.Dynamic.Vehicle.Propulsion.THRUST])
        p.setup(force_alloc_complex=True)
        p.run_model()

    def test_results(self):
        p = self.p
        Dtip_expected = 2.3
        Dtip_units = av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["units"]
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER, Dtip_units)
        np.testing.assert_almost_equal(Dtip_actual, Dtip_expected)

        Rtip_expected = 0.5*2.3
        Rtip_actual = p.get_val("Rtip", units="ft")
        np.testing.assert_almost_equal(Rtip_actual, Rtip_expected)

        thrust_units = av.CoreMetaData[av.Dynamic.Vehicle.Propulsion.THRUST]["units"] 
        thrust_expected = 3*Dtip_expected**2
        thrust_actual = p.get_val(av.Dynamic.Vehicle.Propulsion.THRUST, units=thrust_units)
        np.testing.assert_almost_equal(thrust_actual, thrust_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        partials = cpd["test_comp"]
        np.testing.assert_almost_equal(actual=partials["Rtip", av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=0.5, decimal=12)
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER)
        np.testing.assert_almost_equal(actual=partials[av.Dynamic.Vehicle.Propulsion.THRUST, av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=[6*Dtip_actual], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestAviaryMatrixFreeOutOfPlacePropellerRadiusComp(unittest.TestCase):

    def setUp(self):
        av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["default_value"] = 2.3

        p = self.p = om.Problem()
        jlcomp = jl.get_aviary_matrix_free_test_comp(
            in_place=False,
            aviary_input_names=to_jlsymstrdict({"Dtip": av.Aircraft.Engine.Propeller.DIAMETER}),
            aviary_output_names=to_jlsymstrdict({"thrust": av.Dynamic.Vehicle.Propulsion.THRUST}),
            aviary_meta_data=av.CoreMetaData)

        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=[av.Aircraft.Engine.Propeller.DIAMETER], promotes_outputs=["Rtip", av.Dynamic.Vehicle.Propulsion.THRUST])
        p.setup(force_alloc_complex=True)
        p.run_model()

    def test_results(self):
        p = self.p
        Dtip_expected = 2.3
        Dtip_units = av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["units"]
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER, Dtip_units)
        np.testing.assert_almost_equal(Dtip_actual, Dtip_expected)

        Rtip_expected = 0.5*2.3
        Rtip_actual = p.get_val("Rtip", units="ft")
        np.testing.assert_almost_equal(Rtip_actual, Rtip_expected)

        thrust_units = av.CoreMetaData[av.Dynamic.Vehicle.Propulsion.THRUST]["units"] 
        thrust_expected = 3*Dtip_expected**2
        thrust_actual = p.get_val(av.Dynamic.Vehicle.Propulsion.THRUST, units=thrust_units)
        np.testing.assert_almost_equal(thrust_actual, thrust_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        partials = cpd["test_comp"]
        np.testing.assert_almost_equal(actual=partials["Rtip", av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=0.5, decimal=12)
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER)
        np.testing.assert_almost_equal(actual=partials[av.Dynamic.Vehicle.Propulsion.THRUST, av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=[6*Dtip_actual], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestAviarySparseInPlacePropellerRadiusComp(unittest.TestCase):

    def setUp(self):
        av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["default_value"] = 2.3

        p = self.p = om.Problem()
        jlcomp = jl.get_aviary_sparse_test_comp(
            in_place=True,
            aviary_input_names=to_jlsymstrdict({"Dtip": av.Aircraft.Engine.Propeller.DIAMETER}),
            aviary_output_names=to_jlsymstrdict({"thrust": av.Dynamic.Vehicle.Propulsion.THRUST}),
            aviary_meta_data=av.CoreMetaData)

        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=[av.Aircraft.Engine.Propeller.DIAMETER], promotes_outputs=["Rtip", av.Dynamic.Vehicle.Propulsion.THRUST])
        p.setup(force_alloc_complex=True)
        p.run_model()

    def test_results(self):
        p = self.p
        Dtip_expected = 2.3
        Dtip_units = av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["units"]
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER, Dtip_units)
        np.testing.assert_almost_equal(Dtip_actual, Dtip_expected)

        Rtip_expected = 0.5*2.3
        Rtip_actual = p.get_val("Rtip", units="ft")
        np.testing.assert_almost_equal(Rtip_actual, Rtip_expected)

        thrust_units = av.CoreMetaData[av.Dynamic.Vehicle.Propulsion.THRUST]["units"] 
        thrust_expected = 3*Dtip_expected**2
        thrust_actual = p.get_val(av.Dynamic.Vehicle.Propulsion.THRUST, units=thrust_units)
        np.testing.assert_almost_equal(thrust_actual, thrust_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        partials = cpd["test_comp"]
        np.testing.assert_almost_equal(actual=partials["Rtip", av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=0.5, decimal=12)
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER)
        np.testing.assert_almost_equal(actual=partials[av.Dynamic.Vehicle.Propulsion.THRUST, av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=[6*Dtip_actual], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestAviarySparseOutOfPlacePropellerRadiusComp(unittest.TestCase):

    def setUp(self):
        av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["default_value"] = 2.3

        p = self.p = om.Problem()
        jlcomp = jl.get_aviary_sparse_test_comp(
            in_place=False,
            aviary_input_names=to_jlsymstrdict({"Dtip": av.Aircraft.Engine.Propeller.DIAMETER}),
            aviary_output_names=to_jlsymstrdict({"thrust": av.Dynamic.Vehicle.Propulsion.THRUST}),
            aviary_meta_data=av.CoreMetaData)

        comp = JuliaExplicitComp(jlcomp=jlcomp)
        p.model.add_subsystem("test_comp", comp, promotes_inputs=[av.Aircraft.Engine.Propeller.DIAMETER], promotes_outputs=["Rtip", av.Dynamic.Vehicle.Propulsion.THRUST])
        p.setup(force_alloc_complex=True)
        p.run_model()

    def test_results(self):
        p = self.p
        Dtip_expected = 2.3
        Dtip_units = av.CoreMetaData[av.Aircraft.Engine.Propeller.DIAMETER]["units"]
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER, Dtip_units)
        np.testing.assert_almost_equal(Dtip_actual, Dtip_expected)

        Rtip_expected = 0.5*2.3
        Rtip_actual = p.get_val("Rtip", units="ft")
        np.testing.assert_almost_equal(Rtip_actual, Rtip_expected)

        thrust_units = av.CoreMetaData[av.Dynamic.Vehicle.Propulsion.THRUST]["units"] 
        thrust_expected = 3*Dtip_expected**2
        thrust_actual = p.get_val(av.Dynamic.Vehicle.Propulsion.THRUST, units=thrust_units)
        np.testing.assert_almost_equal(thrust_actual, thrust_expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        # Check that the partials the user provided are correct.
        partials = cpd["test_comp"]
        np.testing.assert_almost_equal(actual=partials["Rtip", av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=0.5, decimal=12)
        Dtip_actual = p.get_val(av.Aircraft.Engine.Propeller.DIAMETER)
        np.testing.assert_almost_equal(actual=partials[av.Dynamic.Vehicle.Propulsion.THRUST, av.Aircraft.Engine.Propeller.DIAMETER]['J_fwd'], desired=[6*Dtip_actual], decimal=12)

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


if __name__ == '__main__':
    unittest.main()
