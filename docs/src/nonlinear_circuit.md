```@meta
CurrentModule = OpenMDAODocs
```
# A More Complicated Example: Nonlinear Circuit
This tutorial will implement [the nonlinear circuit example from the OpenMDAO docs](https://openmdao.org/newdocs/versions/latest/examples/circuit_analysis_examples.html) in Julia.
Along the way, we'll learn

  * how to create implicit components with OpenMDAO.jl
  * how to create OpenMDAO.jl components with metadata (the equivalent of [`options`](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/options.html) in a normal Python component)
  * how to specify default values for component metadata

## Preamble
We'll need the OpenMDAOCore package, of course:

```@example circuit
using OpenMDAOCore: OpenMDAOCore
```

## An Explicit Component with an Option: The `Resistor`
Next we'll create an explicit component that models a resistor.
The resistor has one option: the resistance, `R`.
We'll make `R` a field in the Julia struct that we'll use for the `Resistor` component:

```@example circuit
struct Resistor <: OpenMDAOCore.AbstractExplicitComp
    R::Float64
end
```

Now we'd like to use a default value of `1.0` for the resistance.
We can do that by creating an [outer constructor](https://docs.julialang.org/en/v1/manual/constructors/) for the `Resistor` `struct` with a default keyword value.

```@example circuit
# Default value for R.
Resistor(; R=1.0) = Resistor(R)
```

Next, we'll create a `setup` function as usual:

```@example circuit
function OpenMDAOCore.setup(self::Resistor)
    input_data = [OpenMDAOCore.VarData("V_in"; units="V"), OpenMDAOCore.VarData("V_out"; units="V")]
    output_data = [OpenMDAOCore.VarData("I"; units="A")]

    R = self.R
    partials_data = [OpenMDAOCore.PartialsData("I", "V_in", val=1/R),
                     OpenMDAOCore.PartialsData("I", "V_out", val=-1/R),]

    return input_data, output_data, partials_data
end
```

Since this is a linear resistor, the derivatives are constant, and we can set them via the `val` argument in the `PartialsData` `struct`, just like in OpenMDAO's [`declare_partials`](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_derivatives/specifying_partials.html) method.
Also notice that we can specify units for each of the inputs and outputs, just like in a Python OpenMDAO component.

Finally, we'll implement the `compute!` method:

```@example circuit
function OpenMDAOCore.compute!(self::Resistor, inputs, outputs)
    deltaV = inputs["V_in"][1] - inputs["V_out"][1]
    outputs["I"][1] = deltaV/self.R

    return nothing
end
```

Notice that we have access to the `R` field in the `Resistor` `struct`, named `self` here.
(We could call it anything, just like in a Python method.)

## An Explicit Component with Two Options: The `Diode`
We need two parameters to characterize the Diode: the saturation current `Is` and the thermal voltage `Vt`:

```@example circuit
struct Diode <: OpenMDAOCore.AbstractExplicitComp
    Is::Float64
    Vt::Float64
end
```

Next, we'll create an constructor that sets both options using keyword arguments, and provides default values for both.

```@example circuit
# Use Julia's keyword arguments to set default values.
Diode(; Is=1e-15, Vt=0.025875) = Diode(Is, Vt)
```

Next, we'll implement the `setup` method for the Diode.
```@example circuit
function OpenMDAOCore.setup(self::Diode)
    input_data = [OpenMDAOCore.VarData("V_in"; units="V"), OpenMDAOCore.VarData("V_out"; units="V")]
    output_data = [OpenMDAOCore.VarData("I"; units="A")]

    partials_data = [OpenMDAOCore.PartialsData("I", "V_in"), OpenMDAOCore.PartialsData("I", "V_out")]

    return input_data, output_data, partials_data
end
```
Nothing unusual here.

Finally, the `compute!` and `compute_partials!` methods:

```@example circuit
function OpenMDAOCore.compute!(self::Diode, inputs, outputs)
    deltaV = inputs["V_in"][1] - inputs["V_out"][1]
	Is = self.Is
	Vt = self.Vt
    outputs["I"][1] = Is * (exp(deltaV / Vt) - 1)
    return nothing
end

function OpenMDAOCore.compute_partials!(self::Diode, inputs, J)
	deltaV = inputs["V_in"][1] - inputs["V_out"][1]
	Is = self.Is
	Vt = self.Vt
	I = Is * exp(deltaV / Vt)

	J["I", "V_in"][1, 1] = I/Vt
	J["I", "V_out"][1, 1] = -I/Vt

	return nothing
end
```

Like the `Resistor`, we have access to the `Is` and `Vt` options in the `Diode` `struct` in both methods.

## Our First Implicit Component: The `Node`
Our final component we need for the circuit is an implicit one: the `Node`.
Each node can have an arbitrary number of incoming and outgoing currents, so we'll need two integer options to keep track of that:

```@example circuit
struct Node <: OpenMDAOCore.AbstractImplicitComp
    n_in::Int
	n_out::Int
end
```

We'll have `n_in` and `n_out` both default to one, though:

```@example circuit
Node(; n_in=1, n_out=1) = Node(n_in, n_out)
```

Next up is the `setup` method.
We'll need to use a loop to create all the inputs and outputs needed for the component:

```@example circuit
function OpenMDAOCore.setup(self::Node)
    output_data = [OpenMDAOCore.VarData("V", val=5.0, units="V")]

    input_data = Vector{OpenMDAOCore.VarData}()
    partials_data = Vector{OpenMDAOCore.PartialsData}()

	for i in 0:self.n_in-1
        i_name = "I_in:$i"
        push!(input_data, OpenMDAOCore.VarData(i_name; units="A"))
        push!(partials_data, OpenMDAOCore.PartialsData("V", i_name; val=1.0))
	end

	for i in 0:self.n_out-1
        i_name = "I_out:$i"
        push!(input_data, OpenMDAOCore.VarData(i_name; units="A"))
        push!(partials_data, OpenMDAOCore.PartialsData("V", i_name; val=-1.0))
	end

	return input_data, output_data, partials_data
end
```

We could have done something fancier like an array comprehension to create the `VarData` and `PartialsData` `structs`:

```julia
input_data = [OpenMDAOCore.VarData("I_in:$i"; units="A") for i in 0:self.n_in-1]
```

Also, the derivatives are constant for the node, so we set them in the `PartialsData` `struct`.

Finally, we just need to write the `apply_nonlinear!` method:

```@example circuit
function OpenMDAOCore.apply_nonlinear!(self::Node, inputs, outputs, residuals)
    residuals["V"][1] = 0.0
    for i_conn in 0:self.n_in-1
        residuals["V"][1] += inputs["I_in:$i_conn"][1]
    end
    for i_conn in 0:self.n_out-1
        residuals["V"][1] -= inputs["I_out:$i_conn"][1]
    end

    return nothing
end
```

We see that the `apply_nonlinear!` OpenMDAO.jl method is very similar to the `apply_nonlinear` method on a normal Python [`ImplicitComponent`](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/implicit_component.html#implicitcomponent-methods)— its job is to calculate the residual of the implicit equation(s) it is modeling from the `inputs` and `outputs`.

## The Run Script
We're finally ready for the run script!
Here it is:

```@example circuit
using OpenMDAO: om, make_component

p = om.Problem()

circuit = om.Group()

circuit.add_subsystem("n1", make_component(Node(n_in=1, n_out=2)), promotes_inputs=[("I_in:0", "I_in")])
circuit.add_subsystem("n2", make_component(Node()))

circuit.add_subsystem("R1", make_component(Resistor(R=100.0)), promotes_inputs=[("V_out", "Vg")])
circuit.add_subsystem("R2", make_component(Resistor(R=10000.0)))
circuit.add_subsystem("D1", make_component(Diode()), promotes_inputs=[("V_out", "Vg")])

circuit.connect("n1.V", ["R1.V_in", "R2.V_in"])
circuit.connect("R1.I", "n1.I_out:0")
circuit.connect("R2.I", "n1.I_out:1")

circuit.connect("n2.V", ["R2.V_out", "D1.V_in"])
circuit.connect("R2.I", "n2.I_in:0")
circuit.connect("D1.I", "n2.I_out:0")

circuit.nonlinear_solver = om.NewtonSolver()
circuit.linear_solver = om.DirectSolver()

circuit.nonlinear_solver.options["iprint"] = 2
circuit.nonlinear_solver.options["maxiter"] = 10
circuit.nonlinear_solver.options["solve_subsystems"] = true
circuit.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
circuit.nonlinear_solver.linesearch.options["maxiter"] = 10
circuit.nonlinear_solver.linesearch.options["iprint"] = 2

p.model.add_subsystem("circuit", circuit)

p.setup()

p.set_val("circuit.I_in", 0.1)
p.set_val("circuit.Vg", 0.)

# set some initial guesses
p.set_val("circuit.n1.V", 10.)
p.set_val("circuit.n2.V", 1e-3)

p.run_model()

println("circuit.n1.V = $(p["circuit.n1.V"]) (should be 9.90804735)")
println("circuit.n2.V = $(p["circuit.n2.V"]) (should be 0.71278185)")
println("circuit.R1.I = $(p["circuit.R1.I"]) (should be 0.09908047)")
println("circuit.R2.I = $(p["circuit.R2.I"]) (should be 0.00091953)")
println("circuit.D1.I = $(p["circuit.D1.I"]) (should be 0.00091953)")

# sanity check: should sum to .1 Amps
println("circuit.R1.I + circuit.D1.I = $(p["circuit.R1.I"] + p["circuit.D1.I"]) (should be 0.1)")
```

Notice that:

  * We can use `Groups` and `connect` methods just like a Python OpenMDAO program
  * Linear and nonlinear solvers, and linesearch objects also work fine
  * We get the same answers as the Python example in the OpenMDAO docs! :-)
