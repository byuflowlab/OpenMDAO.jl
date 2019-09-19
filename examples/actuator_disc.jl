using OpenMDAO
using PyCall

julia_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

struct ActuatorDisc <: OpenMDAO.OpenMDAOComp
    options::Dict{AbstractString, Any}
    inputs::AbstractVector{VarData}
    outputs::AbstractVector{VarData}
    partials::AbstractVector{PartialsData}
end

ActuatorDisc() = ActuatorDisc(Dict{String, Any}(), VarData[], VarData[], PartialsData[])

function OpenMDAO.setup!(self::ActuatorDisc)
    push!(self.inputs, VarData("a", [1], [0.5]))
    push!(self.inputs, VarData("Area", [1], [10.0]))
    push!(self.inputs, VarData("rho", [1], [1.225]))
    push!(self.inputs, VarData("Vu", [1], [10.0]))

    push!(self.outputs, VarData("Vr", [1], [0.0]))
    push!(self.outputs, VarData("Vd", [1], [0.0]))
    push!(self.outputs, VarData("Ct", [1], [0.0]))
    push!(self.outputs, VarData("thrust", [1], [0.0]))
    push!(self.outputs, VarData("Cp", [1], [0.0]))
    push!(self.outputs, VarData("power", [1], [0.0]))

    push!(self.partials, PartialsData("Vr", "a"))
    push!(self.partials, PartialsData("Vr", "Vu"))
    push!(self.partials, PartialsData("Vd", "a"))
    push!(self.partials, PartialsData("Ct", "a"))
    push!(self.partials, PartialsData("thrust", "a"))
    push!(self.partials, PartialsData("thrust", "Area"))
    push!(self.partials, PartialsData("thrust", "rho"))
    push!(self.partials, PartialsData("thrust", "Vu"))
    push!(self.partials, PartialsData("Cp", "a"))
    push!(self.partials, PartialsData("power", "a"))
    push!(self.partials, PartialsData("power", "Area"))
    push!(self.partials, PartialsData("power", "rho"))
    push!(self.partials, PartialsData("power", "Vu"))

    return nothing
end

function OpenMDAO.compute!(self::ActuatorDisc, inputs, outputs)
    a = inputs["a"]
    Vu = inputs["Vu"]

    qA = @. .5 * inputs["rho"] * inputs["Area"] * Vu ^ 2

    @. outputs["Vd"] = Vu * (1 - 2 * a)
    Vd = outputs["Vd"]
    @. outputs["Vr"] = .5 * (Vu + Vd)

    @. outputs["Ct"] = 4 * a * (1 - a)
    Ct = outputs["Ct"]
    @. outputs["thrust"] = Ct * qA

    @. outputs["Cp"] = Ct * (1 - a)
    Cp = outputs["Cp"]
    @. outputs["power"] = Cp * qA * Vu
end

function OpenMDAO.compute_partials!(self::ActuatorDisc, inputs, J)
        a = inputs["a"]
        Vu = inputs["Vu"]
        Area = inputs["Area"]
        rho = inputs["rho"]

        # pre-compute commonly needed quantities
        a_times_area = @. a * Area
        one_minus_a = @. 1.0 - a
        a_area_rho_vu = @. a_times_area * rho * Vu

        @. J["Vr", "a"] = -Vu
        @. J["Vr", "Vu"] = one_minus_a

        @. J["Vd", "a"] = -2.0 * Vu

        @. J["Ct", "a"] = 4.0 - 8.0 * a

        @. J["thrust", "a"] = .5 * rho * Vu^2 * Area * J["Ct", "a"]
        @. J["thrust", "Area"] = 2.0 * Vu^2 * a * rho * one_minus_a
        @. J["thrust", "rho"] = 2.0 * a_times_area * Vu ^ 2 * (one_minus_a)
        @. J["thrust", "Vu"] = 4.0 * a_area_rho_vu * (one_minus_a)

        @. J["Cp", "a"] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * (one_minus_a)^2

        @. J["power", "a"] = 2.0 * Area * Vu^3 * a * rho * ( 2.0 * a - 2.0) + 2.0 * Area * Vu^3 * rho * one_minus_a ^ 2
        @. J["power", "Area"] = 2.0 * Vu^3 * a * rho * one_minus_a ^ 2
        @. J["power", "rho"] = 2.0 * a_times_area * Vu ^ 3 * (one_minus_a)^2
        @. J["power", "Vu"] = 6.0 * Area * Vu^2 * a * rho * one_minus_a^2
end

prob = om.Problem()

indeps = om.IndepVarComp()
indeps.add_output("a", 0.5)
indeps.add_output("Area", 10.0)
indeps.add_output("rho", 1.125)
indeps.add_output("Vu", 10.0)
prob.model.add_subsystem("indeps", indeps, promotes=["*"])

actuator_disc = ActuatorDisc()
OpenMDAO.setup!(actuator_disc)
comp = julia_comps.JuliaExplicitComp(julia_comp_data=actuator_disc)
prob.model.add_subsystem("a_disc", comp, promotes_inputs=["a", "Area", "rho", "Vu"])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")

prob.model.add_design_var("a", lower=0., upper=1.)
prob.model.add_design_var("Area", lower=0., upper=1.)

# negative one so we maximize the objective
prob.model.add_objective("a_disc.Cp", scaler=-1)

prob.setup()
prob.run_driver()

# minimum value
println("a_disc.Cp = $(prob.get_val("a_disc.Cp")) (should be 0.59259259)")
println("a = $(prob.get_val("a")) (should be 0.33335528)")
