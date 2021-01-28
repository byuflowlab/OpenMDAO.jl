import gc
import unittest
import openmdao.api as om
from julia.OpenMDAO import make_component
import julia.Main as julia
from julia.Base.GC import gc as gcjl
from julia.OpenMDAO import component_registry_length

import os
d = os.path.dirname(os.path.realpath(__file__))

julia.include(f"{d}/../../../examples/components/simple_explicit.jl")
julia.include(f"{d}/../../../examples/components/simple_implicit.jl")


def main(a):
    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("x", 2.0)
    ivc.add_output("y", 3.0)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    comp = make_component(julia.SimpleExplicit(a))
    prob.model.add_subsystem("square_it_comp0", comp,
                             promotes_inputs=['x', 'y'],
                             promotes_outputs=[('z1', 'z1_0'), ('z2', 'z2_0')])

    comp = make_component(julia.SimpleExplicit(a+1))
    prob.model.add_subsystem("square_it_comp1", comp,
                             promotes_inputs=['x', 'y'],
                             promotes_outputs=[('z1', 'z1_1'), ('z2', 'z2_1')])

    comp = make_component(julia.SimpleExplicit(a+2))
    prob.model.add_subsystem("square_it_comp2", comp,
                             promotes_inputs=['x', 'y'],
                             promotes_outputs=[('z1', 'z1_2'), ('z2', 'z2_2')])

    comp = make_component(julia.SimpleImplicit(a))
    comp.linear_solver = om.DirectSolver(assemble_jac=True)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True, iprint=0, err_on_non_converge=True)
    prob.model.add_subsystem("square_it_comp3", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_3"), ("z2", "z2_3")])

    comp = make_component(julia.SimpleImplicit(a+1))
    comp.linear_solver = om.DirectSolver(assemble_jac=True)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True, iprint=0, err_on_non_converge=True)
    prob.model.add_subsystem("square_it_comp4", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_4"), ("z2", "z2_4")])

    comp = make_component(julia.SimpleImplicit(a+2))
    comp.linear_solver = om.DirectSolver(assemble_jac=True)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True, iprint=0, err_on_non_converge=True)
    prob.model.add_subsystem("square_it_comp5", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_5"), ("z2", "z2_5")])

    prob.setup()
    prob.run_model()

    prob.get_val("z1_0")
    prob.get_val("z2_0")
    prob.get_val("z1_1")
    prob.get_val("z2_1")
    prob.get_val("z1_2")
    prob.get_val("z2_2")
    prob.compute_totals(of=["z1_0", "z2_0"], wrt=["x", "y"])
    prob.compute_totals(of=["z1_1", "z2_1"], wrt=["x", "y"])
    prob.compute_totals(of=["z1_2", "z2_2"], wrt=["x", "y"])


class TestGCComponentRegistry(unittest.TestCase):

    def test_gc_component_registry(self):

        for i in range(3):
            main(3*i)
            self.assertEqual(component_registry_length(), 6)
            gcjl()
            gc.collect()
            self.assertEqual(component_registry_length(), 0)


if __name__ == "__main__":
    unittest.main()
