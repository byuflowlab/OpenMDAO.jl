using OpenMDAO
using PyCall

struct SquareIt2{TF} <: AbstractImplicitComp
    a::TF
end


function OpenMDAO.setup(::SquareIt2)
 
    inputs = VarData[
        VarData("x", shape=[1], val=[2.0]),
        VarData("y", shape=[1], val=[3.0])
    ]

    outputs = VarData[
        VarData("z1", shape=[1], val=[2.0]),
        VarData("z2", shape=[1], val=[3.0])
    ]

    partials = PartialsData[
        PartialsData("z1", "x"),            
        PartialsData("z1", "y"),            
        PartialsData("z1", "z1"),           
        PartialsData("z2", "x"),            
        PartialsData("z2", "y"),            
        PartialsData("z2", "z2")
    ]

    return inputs, outputs, partials
end

function OpenMDAO.apply_nonlinear!(square::SquareIt2, inputs, outputs, residuals)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

function OpenMDAO.linearize!(square::SquareIt2, inputs, outputs, partials)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = 1.0
    @. partials["z1", "x"] = -2*a*x
    @. partials["z1", "y"] = -2*y

    @. partials["z2", "z2"] = 1.0
    @. partials["z2", "x"] = -a
    @. partials["z2", "y"] = -1.0
end

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SquareIt2(2.0))
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=true, iprint=2, err_on_non_converge=true)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
println(prob.get_val("z1"))
println(prob.get_val("z2"))
println(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
