```@meta
CurrentModule = OpenMDAODocs
```
# Variable Shapes at Runtime: `shape_by_conn`, `copy_shape`, and 

## A Simple Example
OpenMDAO is able to [determine variable shapes at runtime](https://openmdao.org/newdocs/versions/latest/features/experimental/dyn_shapes.html).
In "normal" (aka non-Julian) OpenMDAO, this is done via the `shape_by_conn` and `copy_shape` arguments to the venerable `add_input` and/or `add_output` `Component` methods.
In OpenMDAO.jl, we can provide the `shape_by_conn` and/or `copy_shape` arguments to the `VarData` `struct` constructor to get the same behavior.

We'll show how this works using a simple `ExplicitComponent` that computes ``y = 2*x^2 + 1`` element-wise, where ``x`` and ``y`` are two-dimensional arrays of any (identical) size.

We'll need `OpenMDAOCore` of course, and need to declare our `ExplicitComponent` in the usual way:

```@example shape_by_conn1
using OpenMDAOCore: OpenMDAOCore

struct ECompShapeByConn <: OpenMDAOCore.AbstractExplicitComp end
```

Next we need a `setup` method:

```@example shape_by_conn1
function OpenMDAOCore.setup(self::ECompShapeByConn)
    input_data = [OpenMDAOCore.VarData("x"; shape_by_conn=true)]
    output_data = [OpenMDAOCore.VarData("y"; shape_by_conn=true, copy_shape="x")]

    partials_data = []
    return input_data, output_data, partials_data
end
```

Notice how we provided the `shape_by_conn` argument to the `VarData` `struct` for `x`, and the `shape_by_conn` and `copy_shape` arguments to `y`'s `VarData` `struct`.
This means that the shape of `x` will be determined at runtime by OpenMDAO, and will be set to the shape of whatever output is connected to `x`.
The shape of `y` will be set to that of `x`, since we provided the `copy_shape="x"` argument.
(Also notice how we returned an empty Vector for the `partials_data` output—OpenMDAO.jl always expects `OpenMDAOCore.setup` to return three Vectors, corresponding to `input_data`, `output_data`, and `partials_data`.
But the `partials_data` Vector can be empty if it's not needed.)

Now, the derivative of `y` with respect to `x` will be sparse—the value of an element `y[i,j]` depends on the element `x[i,j]`, and no others.
We can communicate this fact to OpenMDAO through the `rows` and `cols` arguments to `declare_partials` in Python OpenMDAO, or the `PartialsData` `struct` in OpenMDAO.jl.
But how do we do that here, when we don't know the sizes of `x` and `y` in the `setup` method?
The answer is we implement an `OpenMDAOCore.setup_partials` method, which gives us another chance to create more `PartialsData` `structs` after OpenMDAO has figured out what the sizes of all the inputs and outputs are:

```@example shape_by_conn1
function OpenMDAOCore.setup_partials(self::ECompShapeByConn, input_sizes, output_sizes)
    @assert input_sizes["x"] == output_sizes["y"]
    m, n = input_sizes["x"]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=Dict(:i=>m, :j=>n), of_ss=[:i, :j], wrt_ss=[:i, :j])
    partials_data = [OpenMDAOCore.PartialsData("y", "x"; rows=rows, cols=cols)]

    return partials_data
end
```

The `OpenMDAOCore.setup_partials` method will always take an instance of the `OpenMDAOCore.AbstractComp` (called `self` here), and two `Dict`s, both with `String` keys and `NTuple{N, Int}` values.
The keys indicate the name of an input or output variable, and the `NTuple{Int, N}` values are the shapes of each variable.
The first `Dict` holds all the input shapes, and the second `Dict` has all the output shapes.

Now, the job of `setup_partials` is to return a `Vector` of `PartialsData` `structs`.
We'd like to include the `rows` and `cols` arguments to the `PartialsData` `struct` for the derivative of `y` with respect to `x`, but it's a bit tricky, since `x` and `y` are two-dimensional.
Luckily, there is a small utility function provided by OpenMDAOCore.jl called `get_rows_cols` that can help us.

## Sparsity Patterns with `get_rows_cols`
The `get_rows_cols` function uses a symbolic notation to express sparsity patterns in a simple way.
Here's an example that corresponds to our present case.
Let's say `x` and `y` have shape `(2, 3)`.
Then the non-zero index combinations for the derivative of `y` with respect to `x` will be (using zero-based indices, which is what OpenMDAO expects for the `rows` and `cols` arguments):

```
y indices:  x indices:
(0, 0)      (0, 0)
(0, 1)      (0, 1)
(0, 2)      (0, 2)
(1, 0)      (1, 0)
(1, 1)      (1, 1)
(1, 2)      (1, 2)
```

So that table says that the value of `y[0, 0]` depends on `x[0, 0]` only, and the value of `y[1, 0]` depends on `x[1, 0]` only, etc..
But OpenMDAO expects flattened indices for the `rows` and `cols` arguments, not multi-dimensional indices.
So we need to convert the multi-dimensional indices in that table to flattened ones.
`get_rows_cols` does that for you, but if you wanted to do that by hand, what I usually do is think of an array having the same shape as each input or output, with each entry in the array corresponding to the entry's flat index.
So for `x` and y, that would be:

```
x_flat_indices =
[0 1 2;
 3 4 5]

y_flat_indices =
[0 1 2;
 3 4 5]
```

(Remember that Python/NumPy arrays use row-major aka C ordering by default.)
So we can now use those two arrays to translate the `y indices` and `x indices` from multi-dimensional to flat:

```
y indices     x indices
multi, flat:  multi, flat:
(0, 0) 0      (0, 0) 0
(0, 1) 1      (0, 1) 1
(0, 2) 2      (0, 2) 2
(1, 0) 3      (1, 0) 3
(1, 1) 4      (1, 1) 4
(1, 2) 5      (1, 2) 5
```

So the `rows` and `cols` arguments will be

```
rows = [0, 1, 2, 3, 4, 5]
cols = [0, 1, 2, 3, 4, 5]
```

where `rows` is the flat non-zero indices for `y`, and `cols` is the flat non-zero indices for `x`.

Now, how do we do this with `get_rows_cols`?
First we have to assign labels to each dimension of `y` and `x`.
The labels must be `Symbols`, and can be anything (but I usually use index-y things like `:i`, `:j`, `:k`, etc.).
We express the sparsity pattern through the choice of labels.
If we use a label for an output dimension that is also used for an input dimension, then we are saying that, for a given index `i` in the "shared" dimension, the value of the output at that index `i` depends on the value of the input index `i` along the labeled dimension, and no others.
For example, if we had a one-dimensional `y` that was calculated from a one-dimensional `x` in this way:

```
for i in 1:10
    y[i] = sin(x[i])
end
```

then we would use the same label for the (single) output and input dimension.

For the present example, we could assign `i` and `j` (say) to the first and second dimensions, respectively, of both `y` and `x`, since `y[i,j]` only depends on `x[i,j]` for all valid `i` and `j`.
We call these `of_ss` (short for "of subscripts for the output) and `wrt_ss` ("with respect to subscripts").

```@example shape_by_conn1
of_ss = [:i, :j]
wrt_ss = [:i, :j]
```

After deciding on the dimension labels, the only other thing we need to do is create a `Dict` that maps the dimension labels to their sizes:

```@example shape_by_conn1
ss_sizes = Dict(:i=>2, :j=>3)
```

since, in our example, the first dimension of `x` and `y` has size `2`, and the second, `3`.

Then we pass those three things to `get_rows_cols`, which then returns the `rows` and `cols` we want.

```@example shape_by_conn1
rows, cols = OpenMDAOCore.get_rows_cols(; ss_sizes, of_ss, wrt_ss)
```

## Back to the Simple Example
Now, back to the simple example.
Remember, we're trying to compute `y = 2*x^2 + 1` elementwise for a 2D `x` and `y`.
The `compute!` method is pretty straight-forward:

```@example shape_by_conn1
function OpenMDAOCore.compute!(self::ECompShapeByConn, inputs, outputs)
    x = inputs["x"]
    y = outputs["y"]
    y .= 2 .* x.^2 .+ 1
    return nothing
end
```

Now, for the `compute_partials!` method, we have to be a bit tricky about the shape of the Jacobian of `y` with respect to `x`.
The `get_rows_cols` function orders the `rows` and `cols` in such a way that the Jacobian gets allocated by OpenMDAO with shape (`i`, `j`), and is then flattened.
Since NumPy arrays are row-major ordered, then, we need to reshape the Jacobian in the opposite order, then switch the dimensions.
This is optional, but makes things easier:

```@example shape_by_conn1
function OpenMDAOCore.compute_partials!(self::ECompShapeByConn, inputs, partials)
    x = inputs["x"]
    m, n = size(x)
    # So, with the way I've declared the partials above, OpenMDAO will have
    # created a Numpy array of shape (m, n) and then flattened it. So, to get
    # that to work, I'll need to do this:
    dydx = PermutedDimsArray(reshape(partials["y", "x"], n, m), (2, 1))
    dydx .= 4 .* x
    return nothing
end
```

## Checking
Now, let's actually create a `Problem` with the new `Component`, along with an `IndepVarComp` that will actually decide on the size:

```@example shape_by_conn1
using OpenMDAO, PythonCall

m, n = 3, 4
p = om.Problem()
comp = om.IndepVarComp()
comp.add_output("x", shape=(m, n))
p.model.add_subsystem("inputs_comp", comp, promotes_outputs=["x"])

ecomp = ECompShapeByConn()
comp = make_component(ecomp)
p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
p.setup(force_alloc_complex=true)
```

Now we should be able to check that the output we get is correct:

```@example shape_by_conn1
p.set_val("x", 1:m*n)
p.run_model()

# Test that the output is what we expect.
expected = 2 .* PyArray(p.get_val("x")).^2 .+ 1
actual = PyArray(p.get_val("y"))
println("expected = $(expected)")
println("actual   = $(actual)")
```

And we can check the derivatives:

```@example shape_by_conn1
p.check_partials(method="cs")
nothing
```

Looks good!
