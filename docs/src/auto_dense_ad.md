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
(The same user-defined function cane be used for the sparse AD and matrix-free AD approaches, also, making it relatively simple to try all three out to see what's fastest!)

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
