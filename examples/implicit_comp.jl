using OpenMDAO
using PyCall

julia_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

struct SquareIt <: OpenMDAO.OpenMDAOComp
    options::Dict{AbstractString, Any}
    inputs::AbstractVector{VarData}
    outputs::AbstractVector{VarData}
    partials::AbstractVector{PartialsData}
end

SquareIt() = SquareIt(Dict{String, Any}(), VarData[], VarData[], PartialsData[])

function OpenMDAO.setup!(self::SquareIt; a=2.0)
    self.options["a"] = a

    push!(self.inputs, VarData("x", [1], [2.0]))
    push!(self.inputs, VarData("y", [1], [3.0]))

    push!(self.outputs, VarData("z1", [1], [2.0]))
    push!(self.outputs, VarData("z2", [1], [3.0]))

    push!(self.partials, PartialsData("z1", "x"))
    push!(self.partials, PartialsData("z1", "y"))
    push!(self.partials, PartialsData("z1", "z1"))
    push!(self.partials, PartialsData("z2", "x"))
    push!(self.partials, PartialsData("z2", "y"))
    push!(self.partials, PartialsData("z2", "z2"))

    return nothing
end

function OpenMDAO.apply_nonlinear!(self::SquareIt, inputs, outputs, residuals)
    a = self.options["a"]
    x = inputs["x"]
    y = inputs["y"]
    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

function OpenMDAO.linearize!(self::SquareIt, inputs, outputs, partials)
    a = self.options["a"]
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

square_it = SquareIt()
OpenMDAO.setup!(square_it)
comp = julia_comps.JuliaImplicitComp(julia_comp_data=square_it)
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=true, iprint=2, err_on_non_converge=true)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
