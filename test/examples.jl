module ExamplesTests

using Test
using OpenMDAO

@testset "simple explicit check" begin
    
    include("../examples/components/simple_explicit.jl")


    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("x", 2.0)
    ivc.add_output("y", 3.0)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    comp = make_component(SimpleExplicit(4.0))  # Need to convert Julia obj to Python obj
    prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

    prob.setup()
    prob.run_model()
    @test all(prob.get_val("z1") .≈ [25.0])
    @test all(prob.get_val("z2") .≈ [11.0])

    deriv = prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
    @test all(deriv["z1", "x"] .≈ [16.0])
    @test all(deriv["z1", "y"] .≈ [6.0])
    @test all(deriv["z2", "x"] .≈ [4.0])
    @test all(deriv["z2", "y"] .≈ [1.0])
end

@testset "simple implicit check" begin

    include("../examples/components/simple_implicit.jl")

    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("x", 2.0)
    ivc.add_output("y", 2.0)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    comp = make_component(SimpleImplicit(2.0))
    comp.linear_solver = om.DirectSolver(assemble_jac=true)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=true, iprint=0, err_on_non_converge=true)
    prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

    prob.setup()
    prob.final_setup()
    prob.run_model()

    @test all(prob.get_val("z1") .≈ [12.0])
    @test all(prob.get_val("z2") .≈ [6.0])

    deriv = prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
    @test all(deriv["z1", "x"] .≈ [8.0])
    @test all(deriv["z1", "y"] .≈ [4.0])
    @test all(deriv["z2", "x"] .≈ [2.0])
    @test all(deriv["z2", "y"] .≈ [1.0])
end

@testset "actuator disc example check" begin

    include("../examples/components/actuator_disc.jl")

    prob = om.Problem()

    indeps = om.IndepVarComp()
    indeps.add_output("a", 0.5)
    indeps.add_output("Area", 10.0)
    indeps.add_output("rho", 1.125)
    indeps.add_output("Vu", 10.0)
    prob.model.add_subsystem("indeps", indeps, promotes=["*"])

    comp = make_component(ActuatorDisc())
    prob.model.add_subsystem("a_disc", comp, promotes_inputs=["a", "Area", "rho", "Vu"])

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", disp=false)

    prob.model.add_design_var("a", lower=0., upper=1.)
    prob.model.add_design_var("Area", lower=0., upper=1.)

    # negative one so we maximize the objective
    prob.model.add_objective("a_disc.Cp", scaler=-1)

    prob.setup()
    prob.run_driver()

    # minimum value
    @test all(prob.get_val("a_disc.Cp") .≈ [0.59259259])
    @test all(prob.get_val("a") .≈ [0.33335528])
end

@testset "implicit comp with apply_linear!" begin

    include("../examples/components/implicit_with_apply_linear.jl")

    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("a", 1.0)
    ivc.add_output("b", -4.0)
    ivc.add_output("c", 3.0)
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    comp = make_component(ImplicitWithApplyLinear())
    comp.linear_solver = om.DirectSolver(assemble_jac=false)
    comp.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=true, iprint=0, err_on_non_converge=true)
    prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

    prob.setup()
    prob.final_setup()
    prob.run_model()
    @test all((prob.get_val("x") .≈ [3.0]))
    derivs = prob.compute_totals(of=["x"], wrt=["a", "b", "c"])
    @test all(derivs["x", "a"] .≈ [-4.5])
    @test all(derivs["x", "b"] .≈ [-1.5])
    @test all(derivs["x", "c"] .≈ [-0.5])
end

end # module
