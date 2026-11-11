```@meta
CurrentModule = OpenMDAODocs
```

# Automatic Dense AD
OpenMDAOCore.jl can create explicit components that are differentiated automatically by the AD packages supported by [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl).
Three approaches to differentiating the components are supported:

* Dense AD, where we assume the Jacobian is dense
* Sparse AD, where we assume the Jacobian is sparse, and use the sparsity pattern to accelerate the differentiation
* Matrix-free AD, where we perform Jacobian-vector or vector-Jacobian products instead of forming the entire Jacobian

This page will describe the first and simplest approach (dense AD), with the next couple of pages getting into the others.

## The User-Defined Function
The interface for the AD functionality in OpenMDAO.jl is a bit different from the "plain" `AbstractExplicitComp` and `AbstractImplicitComp` `struct`s described in earlier examples (see [A Simple Example: Optimizing a Paraboloid](@ref) or [A More Complicated Example: Nonlinear Circuit](@ref)).
Instead of creating subtypes of `AbstractExplicitComp` that implement `OpenMDAOCore.setup`, `OpenMDAOCore.compute!`, etc., we'll be writing a Julia function that performs our desired computation.
This user-defined function will then be passed to a constructor of the `ADExplicitComp (DenseFlavor)` `struct`, which will implement the necessary `OpenMDAOCore` methods for us.
(The same user-defined function can also be used for the sparse AD and matrix-free AD approaches, making it relatively simple to try all three out to see what's fastest!)

The user-defined function *must* follow one of two forms: either it can be an "in-place" function that writes its outputs to an output vector, or it can be an "out-of-place" function that returns a single output vector.
Both types must also have a `params` argument that will contain inputs that are needed for the calculation, but won't be differentiated.
So, an example of an in-place function would be 

```
function f_in_place!(Y, X, params)
   # calculate stuff with X and params, storing result in Y
   return nothing
end
```

where `X` is the input vector and `Y` is the output vector.
(The function doesn't have to return `nothing`, but any returned value will be ignored, so I like to include `return nothing` to make it clear that the return value doesn't matter.)
An out-of-place function would look like

```
function f_out_of_place(X, params)
   # calculate stuff with X and params, returning Y
   return Y
end
```

where again `X` is the input vector.

Now, the `X` and `Y` arguments of those functions must not be plain Julia `Vector`s, but `ComponentVectors` from the [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) package.
What are those?
They are objects provided by the [ComponentArrays.jl](https://github.com/SciML/ComponentArrays.jl) package that act like `Vector`s, but allow the user to define names for each part ("component") of the vector.
For example:

```@example component_vectors
using ComponentArrays: ComponentVector

x1 = ComponentVector(foo=-1.0, bar=-2.0, baz=-3.0)
@show x1 x1[3] x1.foo x1[:foo]
nothing # hide
```

Notice that we can get, say, the third value of `x1` the usual way (`x1[3]`), but also by referring to the `foo` field value via `x1.foo` and by indexing the `ComponentVector` with the symbol `:foo` (`x1[:foo]`).

Each of the components in `x1` are scalars, but they don't have to be:

```@example component_vectors
x2 = ComponentVector(foo=-1.0, bar=1:4, baz=reshape(5:10, 2, 3))
@show x2 x2[:foo] x2[:bar] x2[:baz]
nothing # hide
```

In `x2`, the `foo` component is a scalar, `bar` refers to a `Vector` (aka a 1D `Array`) and `baz` refers to a `Matrix` (aka a 2D Array).
But `x2` still "looks like" a `Vector`:

```@example component_vectors
@show x2[3]  # will give the third value of `x2`, which happens to be the second value of x2[:bar]
@show ndims(x2)  # Should be 1, since a Vector is 1-dimensional
@show length(x2)  # length(x2) gives the total number of entries in `x2`, aka 1 + 4 + 2*3 = 11
@show size(x2)  # size is a length-1 tuple since a Vector has just one dimension
nothing # hide
```

Now, how will we use `ComponentVector`s here?
We'll use them to define the names and sizes of all the inputs and outputs to our component.
For example, with the paraboloid component in [A Simple Example: Optimizing a Paraboloid](@ref), we created one component with two inputs `x` and `y` and one output `f_xy`, all scalars.
So for that case, our `X_ca` would be

```@example component_vectors
X_ca = ComponentVector(x=1.0, y=1.0)
Y_ca = ComponentVector(f_xy=0.0)
@show X_ca Y_ca
nothing # hide
```

Actually, why don't we try to implement the `Paraboloid` component using a `ADExplicitComp (DenseFlavor)`?

## `ADExplicitComp (DenseFlavor)` Paraboloid 
We'll start fresh, first with importing the stuff we'll need:

```@example simple_auto_dense_forwarddiff_paraboloid
using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using OpenMDAO: make_component
```

Next, we need to define the function that implements our paraboloid equation, which, again, is

```math
f(x,y) = (x - 3.0)^2 + xy + (y + 4.0)^2 - 3.0
```

That would look like this:

```@example simple_auto_dense_forwarddiff_paraboloid
function f_paraboloid!(Y_ca, X_ca, params)
    # Get the inputs:
    x = @view(X_ca[:x])
    y = @view(X_ca[:y])
    # Could also do this:
    # x = X_ca.x
    # y = X_ca.y
    # or even this
    # (; x, y) = X_ca

    # Get the output:
    f_xy = @view(Y_ca[:f_xy])
    # Again, could also do this:
    # f_xy = Y_ca.f_xy
    # or
    # (; f_xy) = Y_ca

    # Do the calculation:
    @. f_xy = (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0

    # Return value doesn't matter.
    return nothing
end
nothing # hide
```

  * The `@view` macro is used when extracting the inputs and outputs from the `X_ca` and `Y_ca` `ComponentVector`s.
    This creates a view into the original `ComponentVector`, instead of a new array with a copy of the original data, which avoids unnecessary allocations and (for the outputs) allows modifications to the view to be reflected in the `Y_ca` array.
    In this example everything is a scalar, so no allocations would have happened anyway.
    But it doesn't hurt to use `@view`: it's a good habit to get into, *and* it allows us to use the `@.` [broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting) macro with the scalar `f_xy` output.
  * The `params` argument is not used in this example, but it is still required, since the `ADExplicitComp (DenseFlavor)` constructor will expect the function to accept it.
    Also needed for `ADExplicitComp (SparseFlavor)` and `ADExplicitComp (MatrixFreeFlavor)`.

Our next step is to create the `ComponentVector`s that will be used to hold the inputs and outputs:

```@example simple_auto_dense_forwarddiff_paraboloid
X_ca = ComponentVector(x=1.0, y=1.0)
Y_ca = ComponentVector(f_xy=0.0)
@show X_ca Y_ca
nothing # hide
```

!!! warning "Use sane values for `X_ca` and `Y_ca`"

    The values of the entries in `X_ca` and `Y_ca` will be passed as initial values when creating the OpenMDAO `ExplicitComponent`.
    Depending on your application they may affect e.g. initial guesses for nonlinear solvers or determining the sparsity pattern of your `System`.

Now we're almost ready to create the `ADExplicitComp (DenseFlavor)`.
The last step is to decide what AD library to use.
OpenMDAOCore.jl relies on the [ADTypes.jl](https://github.com/SciML/ADTypes.jl) and DifferentiationInterface.jl packages for implementing the interface for calling the AD.
Theoretically we can use any AD that those packages support.
We'll use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for this example, which is a popular and very robust forward-mode AD:

```@example simple_auto_dense_forwarddiff_paraboloid
using ForwardDiff: ForwardDiff
ad_backend = ADTypes.AutoForwardDiff()
nothing # hide
```

Now we are finally ready to create the component:

```@example simple_auto_dense_forwarddiff_paraboloid
comp = OpenMDAOCore.create_explicit_component(OpenMDAOCore.DenseFlavor(), ad_backend, f_paraboloid!, Y_ca, X_ca; params=nothing)
parab_comp = make_component(comp)
nothing # hide
```

`make_component` will convert the `ADExplicitComp (DenseFlavor)` into a OpenMDAO Python component that we can use with OpenMDAO.
So now we just need to proceed with the paraboloid example as usual:

```@example simple_auto_dense_forwarddiff_paraboloid
using OpenMDAO: om

model = om.Group()
model.add_subsystem("parab_comp", parab_comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("parab_comp.x")
prob.model.add_design_var("parab_comp.y")
prob.model.add_objective("parab_comp.f_xy")

prob.setup(force_alloc_complex=true)

prob.set_val("parab_comp.x", 3.0)
prob.set_val("parab_comp.y", -4.0)

prob.run_model()
println(prob["parab_comp.f_xy"])  # Should print `[-15.]`

prob.set_val("parab_comp.x", 5.0)
prob.set_val("parab_comp.y", -2.0)

prob.run_model()
println(prob.get_val("parab_comp.f_xy"))  # Should print `[-5.]`
nothing # hide
```

Looks OK so far.
But we should check our derivatives, just to be safe.
We can do that with the finite difference method:

```@example simple_auto_dense_forwarddiff_paraboloid
println(prob.check_partials(method="fd"))
nothing # hide
```

or the complex-step method:

```@example simple_auto_dense_forwarddiff_paraboloid
println(prob.check_partials(method="cs"))
nothing # hide
```

Derivatives look great, so let's go ahead and perform the optimization:

```@example simple_auto_dense_forwarddiff_paraboloid
prob.run_driver()
println("f_xy = $(prob.get_val("parab_comp.f_xy"))")  # Should print `[-27.333333]`
println("x = $(prob.get_val("parab_comp.x"))")  # Should print `[6.666666]`
println("y = $(prob.get_val("parab_comp.y"))")  # Should print `[-7.333333]`
nothing # hide
```

Victory!

## An Implicit Component Example: The Nonlinear Circuit

So far, this page has only created *explicit* components.
The same machinery also works for *implicit* components, via the `create_implicit_component` function.
To demonstrate, let's rebuild the nonlinear circuit from [A More Complicated Example: Nonlinear Circuit](@ref).
There, we hand-wrote `setup`, `compute!`, `compute_partials!`, `apply_nonlinear!`, etc. methods for each component.
Here, we'll just write user-defined functions and let the AD machinery generate everything else.

### The User-Defined Functions

The circuit consists of two explicit components (a `Resistor` and a `Diode`) and one implicit component (a `Node`).
The explicit user-defined functions should look familiar:

```@example auto_dense_implicit_circuit
using ComponentArrays: ComponentVector

# Resistor: I = ΔV / R.
# `params.R` plays the role of the `R` option in the hand-written version.
function f_resistor!(Y, X, params)
    Y[:I] = (X[:V_in] - X[:V_out]) / params.R
    return nothing
end

# Diode: I = Is * (exp(ΔV/Vt) - 1).
function f_diode!(Y, X, params)
    deltaV = X[:V_in] - X[:V_out]
    Y[:I] = params.Is * (exp(deltaV / params.Vt) - 1)
    return nothing
end
nothing # hide
```

The implicit function has the same form, except that the first argument is the residual vector `R` instead of the output vector `Y`:

```
f!(R, Y, X, params)  # writes residuals to `R`, using states `Y`, inputs `X`, and parameters `params`
```

The `Node` enforces Kirchhoff's current law: the residual is the sum of the incoming currents minus the sum of the outgoing currents.
We'll give the outgoing currents separate names (`I_out1` and `I_out2`) so they can each be connected to a different source:

```@example auto_dense_implicit_circuit
function f_node!(R, Y, X, params)
    R[:V] = X[:I_in] - X[:I_out1] - X[:I_out2]
    return nothing
end
nothing # hide
```

Notice that the residual doesn't even depend on the state `V` — that's fine!
(The hand-written version's `apply_nonlinear!` was doing exactly the same thing.)

### The Component Vectors

The input and output `ComponentVector`s are the same as before, with one important difference for the implicit component: the values in the state `ComponentVector` `Y_ca` are used as the **initial guesses** for the nonlinear solver.

```@example auto_dense_implicit_circuit
X_ca_R = ComponentVector(V_in=0.0, V_out=0.0)  # resistor inputs
Y_ca_R = ComponentVector(I=0.0)                # resistor output

X_ca_D = ComponentVector(V_in=0.0, V_out=0.0)  # diode inputs
Y_ca_D = ComponentVector(I=0.0)                # diode output

X_ca_N = ComponentVector(I_in=0.0, I_out1=0.0, I_out2=0.0)  # node inputs
Y_ca_N = ComponentVector(V=5.0)                             # node state; 5.0 is the initial guess
nothing # hide
```

### Creating the Components

Now we create the components.
Units are specified with the `units_dict` keyword argument, which maps variable names (as `Symbol`s) to unit strings.
The `Resistor` and `Diode` parameters are passed via the `params` keyword argument, which is available in the user-defined function but is *not* differentiated:

```@example auto_dense_implicit_circuit
using OpenMDAOCore: OpenMDAOCore
using ADTypes: ADTypes
using ForwardDiff: ForwardDiff

ad_backend = ADTypes.AutoForwardDiff()

resistor1_comp = OpenMDAOCore.create_explicit_component(
    OpenMDAOCore.DenseFlavor(), ad_backend, f_resistor!, Y_ca_R, X_ca_R;
    params=(R=100.0,), units_dict=Dict(:V_in=>"V", :V_out=>"V", :I=>"A"))
resistor2_comp = OpenMDAOCore.create_explicit_component(
    OpenMDAOCore.DenseFlavor(), ad_backend, f_resistor!, Y_ca_R, X_ca_R;
    params=(R=10000.0,), units_dict=Dict(:V_in=>"V", :V_out=>"V", :I=>"A"))

diode_comp = OpenMDAOCore.create_explicit_component(
    OpenMDAOCore.DenseFlavor(), ad_backend, f_diode!, Y_ca_D, X_ca_D;
    params=(Is=1e-15, Vt=0.025875), units_dict=Dict(:V_in=>"V", :V_out=>"V", :I=>"A"))

node1_comp = OpenMDAOCore.create_implicit_component(
    OpenMDAOCore.DenseFlavor(), true, ad_backend, f_node!, Y_ca_N, X_ca_N;
    units_dict=Dict(:I_in=>"A", :I_out1=>"A", :I_out2=>"A", :V=>"V"))
node2_comp = OpenMDAOCore.create_implicit_component(
    OpenMDAOCore.DenseFlavor(), true, ad_backend, f_node!, Y_ca_N, X_ca_N;
    units_dict=Dict(:I_in=>"A", :I_out1=>"A", :I_out2=>"A", :V=>"V"))
nothing # hide
```

Note the extra `in_place=true` argument in the `create_implicit_component` calls, indicating that the user-defined function is in-place.
(For an out-of-place function we would pass `in_place=false`.)

The same `f_node!` function is used for both nodes.
Node 2 only has one outgoing current (`D1.I`), so its `I_out2` input is simply left unconnected, in which case OpenMDAO uses its declared value (`0.0`) — contributing nothing to the residual.

Implicit components can also customize the initial state guesses that OpenMDAO's nonlinear solvers use: pass a guess function via the `guess_nonlinear` keyword argument to `create_implicit_component` (see its docstring for the in-place and out-of-place signatures), and OpenMDAO will call it through [`OpenMDAOCore.guess_nonlinear!`](@ref) before each nonlinear solve.

### The Run Script

The run script is identical to the hand-written version:
we build the same model with the same groups, connections, and solvers — the only difference is where the components came from:

```@example auto_dense_implicit_circuit
using OpenMDAO: om, make_component

p = om.Problem()

circuit = om.Group()

circuit.add_subsystem("n1", make_component(node1_comp), promotes_inputs=[("I_in", "I_in")])
circuit.add_subsystem("n2", make_component(node2_comp))

circuit.add_subsystem("R1", make_component(resistor1_comp), promotes_inputs=[("V_out", "Vg")])
circuit.add_subsystem("R2", make_component(resistor2_comp))
circuit.add_subsystem("D1", make_component(diode_comp), promotes_inputs=[("V_out", "Vg")])

circuit.connect("n1.V", ["R1.V_in", "R2.V_in"])
circuit.connect("R1.I", "n1.I_out1")
circuit.connect("R2.I", "n1.I_out2")

circuit.connect("n2.V", ["R2.V_out", "D1.V_in"])
circuit.connect("R2.I", "n2.I_in")
circuit.connect("D1.I", "n2.I_out1")

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

# sanity check: should sum to 0.1 Amps
println("circuit.R1.I + circuit.D1.I = $(p["circuit.R1.I"] + p["circuit.D1.I"]) (should be 0.1)")
```

We get the same answers as the hand-written version — of course!
But notice what we *didn't* have to write: no `struct` definitions, no `setup` methods, no `linearize!` or `compute_partials!` — the `ADImplicitComp (DenseFlavor)` differentiates the residual function automatically, assembles the Jacobian, and hands it to the OpenMDAO solvers.
