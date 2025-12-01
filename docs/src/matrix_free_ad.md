```@meta
CurrentModule = OpenMDAODocs
```

# (Experimental) Automatic Matrix-Free AD
OpenMDAOCore.jl can create explicit components that are differentiated automatically by the AD packages supported by [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) using Jacobian-vector or vector-Jacobian products instead of assembling the complete Jacobian.
The resulting components will use the [OpenMDAO Matrix-Free API](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/explicit_component.html?highlight=matrix%20free#the-matrix-free-api-providing-derivatives-as-a-matrix-vector-product).

## The User-Defined Function for Matrix-Free AD
The requirements for the user-defined function for the matrix-free AD functionality is identical to those for [Automatic Dense AD](@ref) and [Automatic Sparse AD](@ref), and the interface is much the same, as well.
We'll still need to provide a function that expects a `ComponentVector` for its input, and either write its outputs to a `ComponentVector` (for the in-place form) or return a `ComponentVector` with outputs (for the out-of-place form).
The only significant difference is that users will create a `MatrixFreeADExplicitComp` instead of a `SparseADExplicitComp`.

## `MatrixFreeADExplicitComp` Paraboloid 
Let's do the good old Paraboloid example yet again, this time with the `MatrixFreeADExplicitComp`.
We'll load the same packages as we did for the sparse AD example (except we don't need `SparseMatrixColorings` since we won't be doing sparsity):

```@example matrix_free_paraboloid
using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using OpenMDAO: make_component
```

We'll be using the same paraboloid function, but this time let's create an out-of-place function:

```@example matrix_free_paraboloid
function f_paraboloid(X_ca, params)
    # Get the inputs:
    # Using @view with ReverseDiff.jl doesn't work for some reason.
    # x = @view(X_ca[:x])
    # y = @view(X_ca[:y])
    # Could also do this:
    x = X_ca.x
    y = X_ca.y
    # or even this
    # (; x, y) = X_ca

    # Do the calculation:
    f_xy = @. (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0

    return ComponentVector(f_xy=f_xy)
end
nothing # hide
```

(For some reason the `@view` macro doesn't work with ReverseDiff.jl, the AD library we'll use for this example.)

Now we'll create the input `ComponentVector`.
No need to create the output `ComponentVector` for the out-of-place callback function.

```@example matrix_free_paraboloid
X_ca = ComponentVector(x=1.0, y=1.0)
nothing # hide
```

And finally we'll decide which AD library we'll use.
For this one, let's try [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl):

```@example matrix_free_paraboloid
using ReverseDiff: ReverseDiff
ad_backend = ADTypes.AutoReverseDiff()
nothing # hide
```

Now we can create the component:

```@example matrix_free_paraboloid
comp = OpenMDAOCore.MatrixFreeADExplicitComp(ad_backend, f_paraboloid, X_ca)
parab_comp = make_component(comp)
nothing # hide
```

As before, `make_component` will convert the `MatrixFreeADExplicitComp` into a OpenMDAO Python component that we can use with OpenMDAO.
So now we just need to proceed with the paraboloid example as usual.
But!
We need to make sure to [tell OpenMDAO that we need to calculate total derivatives in reverse mode](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_derivatives/picking_mode.html), not forward, since we're using reverse AD for our paraboloid component.
We do that by supplying the `mode="rev"` argument to `Problem.setup()`.

```@example matrix_free_paraboloid
using OpenMDAO: om

prob = om.Problem()

model = om.Group()
model.add_subsystem("parab_comp", parab_comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("parab_comp.x")
prob.model.add_design_var("parab_comp.y")
prob.model.add_objective("parab_comp.f_xy")

prob.setup(force_alloc_complex=true, mode="rev")

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

Looks good.
Let's check our derivatives using finite difference:

```@example matrix_free_paraboloid
println(prob.check_partials(method="fd"))
nothing # hide
```

Now, some of the derivatives don't look so great!
Why?
There's a hint at the beginning of the output above in the form of a warning from OpenMDAOCore.jl.
When checking the partial derivatives of a matrix-free component, OpenMDAO runs the component in both forward and reverse mode, showing the results in the output above as `Jfor` for forward, `Jrev` for reverse (and `Jfd` for the finite difference approximation to the derivatives).
ReverseDiff.jl only supports reverse mode, however, so the derivatives calculated by our paraboloid component will be incorrect when run in forward mode (as the warning message tells us).
Similarly, if we had chosen ForwardDiff.jl, the reverse-mode derivatives would have been incorrect.
So, when looking at the output above, we need to ignore any result that involves `Jfor`, and just look at the comparisons between `Jrev` and `Jfd`.
With that in mind, the derivatives look quite good.

We can also check the derivatives with the finite difference method:

```@example matrix_free_paraboloid
println(prob.check_partials(method="cs"))
nothing # hide
```

Same story: the derivatives look good, assuming we just look at the comparisons between `Jrev` and `Jfd`.
So we're ready to actually run the optimization to verify everything is working properly:

```@example matrix_free_paraboloid
prob.run_driver()
println("f_xy = $(prob.get_val("parab_comp.f_xy"))")  # Should print `[-27.33333333]`
println("x = $(prob.get_val("parab_comp.x"))")  # Should print `[6.66666633]`
println("y = $(prob.get_val("parab_comp.y"))")  # Should print `[-7.33333367]`
nothing # hide
```

We got the right answer, so everything's good!
