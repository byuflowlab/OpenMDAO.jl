using OpenMDAO
using PyCall

om = pyimport("openmdao.api")

struct ActuatorDisc <: OpenMDAO.AbstractExplicitComp
end

function OpenMDAO.setup(self::ActuatorDisc)
    inputs = VarData[]
    push!(inputs, VarData("a", 1, [0.5]))
    push!(inputs, VarData("Area", 1, [10.0]))
    push!(inputs, VarData("rho", 1, [1.225]))
    push!(inputs, VarData("Vu", 1, [10.0]))

    outputs = VarData[]
    push!(outputs, VarData("Vr", 1, [0.0]))
    push!(outputs, VarData("Vd", 1, [0.0]))
    push!(outputs, VarData("Ct", 1, [0.0]))
    push!(outputs, VarData("thrust", 1, [0.0]))
    push!(outputs, VarData("Cp", 1, [0.0]))
    push!(outputs, VarData("power", 1, [0.0]))

    partials = PartialsData[]
    push!(partials, PartialsData("Vr", "a"))
    push!(partials, PartialsData("Vr", "Vu"))
    push!(partials, PartialsData("Vd", "a"))
    push!(partials, PartialsData("Ct", "a"))
    push!(partials, PartialsData("thrust", "a"))
    push!(partials, PartialsData("thrust", "Area"))
    push!(partials, PartialsData("thrust", "rho"))
    push!(partials, PartialsData("thrust", "Vu"))
    push!(partials, PartialsData("Cp", "a"))
    push!(partials, PartialsData("power", "a"))
    push!(partials, PartialsData("power", "Area"))
    push!(partials, PartialsData("power", "rho"))
    push!(partials, PartialsData("power", "Vu"))

    return (inputs, outputs, partials)
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

comp = make_component(ActuatorDisc())
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
