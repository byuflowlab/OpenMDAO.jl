import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import julia.Main as julia
from omjl import make_component

import os
d = os.path.dirname(os.path.realpath(__file__))

julia.include(f"{d}/../../../examples/components/simple_explicit.jl")
julia.include(f"{d}/../../../examples/components/simple_implicit.jl")
julia.include(f"{d}/../../../examples/components/actuator_disc.jl")


class TestExplicitComp(unittest.TestCase):

    def test_default_values(self):
        p = om.Problem(model=make_component(julia.SimpleExplicit(2.0)))

        p.setup()

        assert_rel_error(self, p['x'], 2.0)
        assert_rel_error(self, p['y'], 3.0)
        assert_rel_error(self, p['z1'], 2.0)

    def test_same_data_after_run_model(self):
        p = om.Problem(model=make_component(julia.SimpleExplicit(2.0)))

        p.setup()
        p.final_setup()  # Seems like I need this for the id checks to be the same?

        id_x = id(p['x'])
        id_y = id(p['y'])
        id_z1 = id(p['z1'])
        id_z2 = id(p['z2'])

        p.run_model()

        id_x_check = id(p['x'])
        id_y_check = id(p['y'])
        id_z1_check = id(p['z1'])
        id_z2_check = id(p['z2'])

        self.assertEqual(id_x, id_x_check)
        self.assertEqual(id_y, id_y_check)
        self.assertEqual(id_z1, id_z1_check)
        self.assertEqual(id_z2, id_z2_check)

    def test_same_data_after_compute_totals(self):
        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output("x", 2.0)
        ivc.add_output("y", 3.0)
        p.model.add_subsystem("ivc", ivc, promotes=["*"])

        comp = make_component(julia.SimpleExplicit(4.0))
        p.model.add_subsystem("square_it_comp", comp, promotes=["*"])

        p.setup()
        p.final_setup()  # Seems like I need this for the id checks to be the same?

        id_x = id(p['x'])
        id_y = id(p['y'])
        id_z1 = id(p['z1'])
        id_z2 = id(p['z2'])

        p.run_model()
        p.compute_totals(of=["z1", "z2"], wrt=["x", "y"])

        id_x_check = id(p['x'])
        id_y_check = id(p['y'])
        id_z1_check = id(p['z1'])
        id_z2_check = id(p['z2'])

        self.assertEqual(id_x, id_x_check)
        self.assertEqual(id_y, id_y_check)
        self.assertEqual(id_z1, id_z1_check)
        self.assertEqual(id_z2, id_z2_check)

    # def test_same_data_after_compute(self):
    #     # How do I check that I have the same data after the compute? I guess I need the inputs and outputs.
    #     p = om.Problem()

    #     ivc = om.IndepVarComp()
    #     ivc.add_output("x", 2.0)
    #     ivc.add_output("y", 3.0)
    #     p.model.add_subsystem("ivc", ivc, promotes=["*"])

    #     comp = JuliaExplicitComp(julia_comp_data=SimpleExplicit(4.0))
    #     p.model.add_subsystem("square_it_comp", comp, promotes=["*"])

    #     p.setup()
    #     p.final_setup()  # Seems like I need this for the id checks to be the same?

    #     p.model.list_inputs()
    #     p.model.list_outputs()
    #     print(p.model.square_it_comp)
    #     # print(dir(p.model.square_it_comp))
    #     print(p.model.square_it_comp._inputs['x'])
    #     print(p.model.square_it_comp._inputs['y'])
    #     print(p.model.square_it_comp._outputs['z1'])
    #     print(p.model.square_it_comp._outputs['z2'])

    #     before_ids = {}
    #     for v in ('x', 'y'):
    #     after_ids = {}
    #     comp.compute(comp._inputs, comp._outputs)


class TestImplicitComp(unittest.TestCase):

    def test_default_values(self):
        p = om.Problem(model=make_component(julia.SimpleImplicit(2.0)))

        p.setup()

        assert_rel_error(self, p['x'], 2.0)
        assert_rel_error(self, p['y'], 3.0)
        assert_rel_error(self, p['z1'], 2.0)
        assert_rel_error(self, p['z2'], 3.0)

    def test_same_data_after_run_model(self):
        p = om.Problem(model=make_component(julia.SimpleImplicit(2.0)))

        p.setup()
        p.final_setup()  # Seems like I need this for the id checks to be the same?

        id_x = id(p['x'])
        id_y = id(p['y'])
        id_z1 = id(p['z1'])
        id_z2 = id(p['z2'])

        p.run_model()

        id_x_check = id(p['x'])
        id_y_check = id(p['y'])
        id_z1_check = id(p['z1'])
        id_z2_check = id(p['z2'])

        self.assertEqual(id_x, id_x_check)
        self.assertEqual(id_y, id_y_check)
        self.assertEqual(id_z1, id_z1_check)
        self.assertEqual(id_z2, id_z2_check)

    def test_same_data_after_compute_totals(self):
        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output("x", 2.0)
        ivc.add_output("y", 3.0)
        p.model.add_subsystem("ivc", ivc, promotes=["*"])

        comp = make_component(julia.SimpleImplicit(4.0))
        p.model.add_subsystem("square_it_comp", comp, promotes=["*"])

        p.setup()
        p.final_setup()  # Seems like I need this for the id checks to be the same?

        id_x = id(p['x'])
        id_y = id(p['y'])
        id_z1 = id(p['z1'])
        id_z2 = id(p['z2'])

        p.run_model()
        p.compute_totals(of=["z1", "z2"], wrt=["x", "y"])

        id_x_check = id(p['x'])
        id_y_check = id(p['y'])
        id_z1_check = id(p['z1'])
        id_z2_check = id(p['z2'])

        self.assertEqual(id_x, id_x_check)
        self.assertEqual(id_y, id_y_check)
        self.assertEqual(id_z1, id_z1_check)
        self.assertEqual(id_z2, id_z2_check)


if __name__ == "__main__":
    unittest.main()
