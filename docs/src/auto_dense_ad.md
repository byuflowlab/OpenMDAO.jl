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
This user-defined function will then be passed to a constructor of the `DenseADExplicitComp` `struct`, which will implement the necessary `OpenMDAOCore` methods for us.
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

Actually, why don't we try to implement the `Paraboloid` component using a `DenseADExplicitComp`?

## `DenseADExplicitComp` Paraboloid 
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
  * The `params` argument is not used in this example, but it is still required, since the `DenseADExplicitComp` constructor will expect the function to accept it.
    Also needed for `SparseADExplicitComp` and `MatrixFreeADExplicitComp`.

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

Now we're almost ready to create the `SparseADExplicitComp`.
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
comp = OpenMDAOCore.DenseADExplicitComp(ad_backend, f_paraboloid!, Y_ca, X_ca; params=nothing)
parab_comp = make_component(comp)
nothing # hide
```

`make_component` will convert the `DenseADExplicitComp` into a OpenMDAO Python component that we can use with OpenMDAO.
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
