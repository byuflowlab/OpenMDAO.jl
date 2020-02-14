module GCComponentRegistry

using OpenMDAO: om, make_component, component_registry
using Base.GC
using PyCall
using Test

pygc = pyimport("gc")

include("../examples/components/simple_explicit.jl")
include("../examples/components/simple_implicit.jl")


function main(a)
    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("x", 2.0)
    ivc.add_output("y", 3.0)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    comp = make_component(SimpleExplicit(a))
    prob.model.add_subsystem("square_it_comp0", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_0"), ("z2", "z2_0")])

    comp = make_component(SimpleExplicit(a+1))
    prob.model.add_subsystem("square_it_comp1", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_1"), ("z2", "z2_1")])

    comp = make_component(SimpleExplicit(a+2))
    prob.model.add_subsystem("square_it_comp2", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_2"), ("z2", "z2_2")])

    comp = make_component(SimpleImplicit(a))
    comp.linear_solver = om.DirectSolver(assemble_jac=true)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=true, iprint=0, err_on_non_converge=true)
    prob.model.add_subsystem("square_it_comp3", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_3"), ("z2", "z2_3")])

    comp = make_component(SimpleImplicit(a+1))
    comp.linear_solver = om.DirectSolver(assemble_jac=true)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=true, iprint=0, err_on_non_converge=true)
    prob.model.add_subsystem("square_it_comp4", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_4"), ("z2", "z2_4")])

    comp = make_component(SimpleImplicit(a+2))
    comp.linear_solver = om.DirectSolver(assemble_jac=true)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=true, iprint=0, err_on_non_converge=true)
    prob.model.add_subsystem("square_it_comp5", comp,
                             promotes_inputs=["x", "y"],
                             promotes_outputs=[("z1", "z1_5"), ("z2", "z2_5")])

    prob.setup()
    prob.run_model()

    return nothing
end


@testset "component_registry garbage collection" begin
    GC.gc()
    pygc.collect()
    n = length(component_registry)
    for i in 1:3
        main(3*i)
        @test length(component_registry) == n+6
        GC.gc()
        pygc.collect()
        @test length(component_registry) == n
    end
end


end # module
