```@meta
CurrentModule = OpenMDAODocs
```
# Creating Python Packages That Depend On OpenMDAOCore.jl
The OpenMDAO.jl repository contains an example Python package that implements a couple of the examples described in the previous docs (the [`SparseADExplicitComp` Paraboloid](@ref) and [`SparseADExplicitComp` with Actual Sparsity](@ref) examples) in the `examples/example_python_package_openmdao_jl/` sub-directory.
The package structure looks like this:

```
shell> tree example_python_package_openmdao_jl/
example_python_package_openmdao_jl/
├── example_python_package_openmdao_jl
│   ├── circle.jl
│   ├── circle.py
│   ├── __init__.py
│   ├── julia
│   │   └── JuliaExampleComponents
│   │       ├── Project.toml
│   │       ├── src
│   │       │   ├── circle.jl
│   │       │   ├── JuliaExampleComponents.jl
│   │       │   └── paraboloid.jl
│   │       └── test
│   │           ├── Project.toml
│   │           └── runtests.jl
│   ├── juliapkg.json
│   ├── paraboloid.jl
│   ├── paraboloid.py
│   └── test
│       ├── test_circle_example.py
│       └── test_paraboloid_example.py
├── MANIFEST.in
├── pyproject.toml
└── scripts
    ├── run_circle.py
    └── run_paraboloid.py

8 directories, 23 files

shell> 
```
You can look through that to see how things are put together, but I'll point out a few "best-practices" that I've found.

## First tip: create a Python package that depends on `omjlcomps`
The `example_python_package_openmdao_jl` package is a Python package that uses  the `pyproject.toml` file to declare that it depends on `omjlcomps` in the usual way:

```
shell> cat example_python_package_openmdao_jl/pyproject.toml
[project]
name = 'example_python_package_openmdao_jl'
description = "Example Python package using OpenMDAOCore.jl"
version = "0.1.0"

dependencies = [
    "openmdao~=3.26",
    "juliapkg~=0.1.10",
    "juliacall~=0.9.13",
    "omjlcomps>=0.2.5",
]

shell> 
```

(There are many approaches to creating Python packages.
Any approach should work, as long as the dependencies above are included---your mileage may vary.)

It also depends on `openmdao` (of course), and [`juliapkg`](https://github.com/JuliaPy/pyjuliapkg) and [`juliacall`](https://github.com/JuliaPy/PythonCall.jl).
`juliapkg` is a Python package that allows us to define Julia dependencies of Python packages, and `juliacall` is what we use to call Julia code from Python.
But how will we, specifically, declare which Julia packages we need?
That's what the `juliapkg.json` file is for, which `juliapkg` reads when deciding what Julia packages are needed for a given Python package.
After reading the [`juliapkg` docs on the subject](https://github.com/JuliaPy/pyjuliapkg?tab=readme-ov-file#declare-dependencies), we could edit that file directly, listing out each Julia package that we need in excruciating detail.
A slightly less annoying approach would be to use `juliapkg`'s functional interface, running commands like `juliapkg.add` from the Python REPL.
My preferred approach instead is to have a short `juliapkg.json` file that looks like this:

```
shell> cat example_python_package_openmdao_jl/example_python_package_openmdao_jl/juliapkg.json
{"packages": {
    "JuliaExampleComponents": {"uuid": "4f289198-a466-4861-a6bd-1e7b09ed8707", "dev": true, "path": "./julia/JuliaExampleComponents"}
    }
}

shell> 
```

That says that our example Python package will depend on just one Julia package named `JuliaExampleComponents`, which is located under `julia/JuliaExampleComponents` (relative to the location of the `juliapkg.json` file itself).
The `"dev": true` entry means that the Julia package will be installed in [`dev` mode](https://pkgdocs.julialang.org/v1/managing-packages/#developing), meaning that any changes to the Julia source will be tracked and loaded by Julia package manager (the equivalent of `pip install --editable <path>` or `pip install -e <path>` in Python).
`juliapkg` will automatically install any dependencies needed by the JuliaExampleComponents package.


## Second tip: create a Julia package within the Python source code to easily track Julia dependencies
But, how can we create a Julia package, like JuliaExampleComponents?
The Julia package manager has a command called `generate` that will create a skeleton of a Julia package for you.
Navigate to the location in your Python package where you'd like to store the julia code (I prefer creating a `julia` subdirectory within the Python source), open up the Julia REPL, type `]` to get to the `pkg>` mode, and then run `generate MyPackageName`, where `MyPackageName` is any name you'd like.
For example, the JuliaExampleComponents package was created with something like:

```
(docs) pkg> generate JuliaExampleComponents
  Generating  project JuliaExampleComponents:
    JuliaExampleComponents/Project.toml
    JuliaExampleComponents/src/JuliaExampleComponents.jl

(docs) pkg> 
```

To add dependencies to the Julia package, first `activate` the package from the `pkg>` mode, like this:

```
(docs) pkg> activate JuliaExampleComponents/
  Activating project at `~/desk/foo/JuliaExampleComponents`

(JuliaExampleComponents) pkg> 
```

(Notice the change in the name to the left of the `pkg>` prompt, which tells you the name of the active Julia environment.)
Now, to add a dependency, we just use the `add` command, again within the `pkg>` mode.
For example, the JuliaExampleComponents package will need to use OpenMDAOCore.jl to create OpenMDAO components, so we can add that like so:

```
(JuliaExampleComponents) pkg> add OpenMDAOCore
   Resolving package versions...
      Compat entries added for OpenMDAOCore
    Updating `~/desk/foo/JuliaExampleComponents/Project.toml`
  [24d19c10] + OpenMDAOCore v0.3.2
    Updating `~/desk/foo/JuliaExampleComponents/Manifest.toml`
  [47edcb42] + ADTypes v1.18.0
  [79e6a3ab] + Adapt v4.4.0
  [4fba245c] + ArrayInterface v7.20.0
  [d360d2e6] + ChainRulesCore v1.26.0
  [f70d9fcc] + CommonWorldInvalidations v1.0.0
  [34da2185] + Compat v4.18.0
  [b0b7db55] + ComponentArrays v0.15.29
  [187b0558] + ConstructionBase v1.6.0
⌅ [864edb3b] + DataStructures v0.18.22
⌅ [a0c0ee7d] + DifferentiationInterface v0.6.54
  [ffbed154] + DocStringExtensions v0.9.5
  [d9f16b24] + Functors v0.5.2
  [615f187c] + IfElse v0.1.1
  [24d19c10] + OpenMDAOCore v0.3.2
  [bac558e1] + OrderedCollections v1.8.1
⌅ [aea7be01] + PrecompileTools v1.2.1
  [21216c6a] + Preferences v1.5.0
  [ae029012] + Requires v1.3.1
  [431bcebd] + SciMLPublic v1.0.0
  [0a514795] + SparseMatrixColorings v0.4.21
  [aedffcd0] + Static v1.3.0
  [0d7ed370] + StaticArrayInterface v1.8.0
  [1e83bf80] + StaticArraysCore v1.4.3
  [1986cc42] + Unitful v1.25.0
  [6fb2a4bd] + UnitfulAngles v0.7.2
  [56f22d72] + Artifacts v1.11.0
  [2a0f44e3] + Base64 v1.11.0
  [ade2ca70] + Dates v1.11.0
  [b77e0a4c] + InteractiveUtils v1.11.0
  [8f399da3] + Libdl v1.11.0
  [37e2e46d] + LinearAlgebra v1.11.0
  [d6f4376e] + Markdown v1.11.0
  [de0858da] + Printf v1.11.0
  [9a3f8284] + Random v1.11.0
  [ea8e919c] + SHA v0.7.0
  [9e88b42a] + Serialization v1.11.0
  [2f01184e] + SparseArrays v1.11.0
  [fa267f1f] + TOML v1.0.3
  [cf7118a7] + UUIDs v1.11.0
  [4ec0a83e] + Unicode v1.11.0
  [e66e0078] + CompilerSupportLibraries_jll v1.1.1+0
  [4536629a] + OpenBLAS_jll v0.3.27+1
  [bea87d4a] + SuiteSparse_jll v7.7.0+0
  [8e850b90] + libblastrampoline_jll v5.11.0+0
        Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`

(JuliaExampleComponents) pkg> 
```

We can check what direct dependencies are associated with a particular Julia package/environment by using the `status` command.
Again looking at the JuliaExampleComponents package:

```
(JuliaExampleComponents) pkg> status
Project JuliaExampleComponents v0.1.0
Status `~/desk/foo/JuliaExampleComponents/Project.toml`
  [24d19c10] OpenMDAOCore v0.3.2

(JuliaExampleComponents) pkg> 
```

Source code is can be added to `MyPackageName/src/MyPackageName.jl` file, and the `include` function can be used to organize the code in different files, etc..
Also, optionally, you can use the [usual Julia testing infrastructure](https://pkgdocs.julialang.org/dev/creating-packages/#adding-tests-to-packages) to add tests to the package.

## Third tip: create small "entry point" files for getting the Julia components into Python and OpenMDAO
Now, how do we get the components defined in the Julia package into Python?
Within the Julia package itself, I like to create small functions that take in parameters needed to construct the OpenMDAOCore.jl struct and return the `struct` itself.
For example, the Paraboloid component in `JuliaExampleComponents` has a function called `get_paraboloid_comp` that looks like this:

```
function get_parabaloid_comp()
    X_ca = ComponentVector(x=1.0, y=1.0)
    Y_ca = ComponentVector(f_xy=0.0)

    # Absolute tolerance for detecting zero entries in the Jacobian.
    sparse_atol = 1e-10
    # Use ForwardDiff.jl to calculate the Jacobian and determine sparsity, evaluating the Jacobian multiple times with a perturbed input vector.
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    # Coloring algorithm for determining the color of the Jacobian.
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    # AD library: will also use ForwardDiff.jl to calculate the sparse jacobian each time.
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    # Create the OpenMDAOCore.jl component.
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_paraboloid!, Y_ca, X_ca; params=nothing)

    return comp
end
```

That function takes no arguments, but one could imagine passing in things that control what AD library the component will use, parameters associated with the size of the inputs and outputs, etc..

Now, to get that in Julia, I first create a small Julia file that just imports the function we wrote to get the OpenMDAOCore.jl `struct`.
For example, in `example_python_package_openmdao_jl` we have a file called `paraboloid.jl` that looks like this:

```
shell> cat example_python_package_openmdao_jl/example_python_package_openmdao_jl/paraboloid.jl
using JuliaExampleComponents: get_parabaloid_comp

shell> 
```

Then, in the same directory, we create a Python file that uses `juliacall` to include that Julia file, and thus access `get_paraboloid_comp` from Python.
So, again for the paraboloid component, we have a file called `paraboloid.py` that looks like this

```
shell> cat example_python_package_openmdao_jl/example_python_package_openmdao_jl/paraboloid.py
import os

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall; jl = juliacall.newmodule("ParaboloidComponentsStub")

# Get the directory this file is in, then include the `paraboloid.jl` Julia source code.
d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "paraboloid.jl"))
# Now we have access to everything in `paraboloid.jl` in the `jl` object.

get_parabaloid_comp = jl.get_parabaloid_comp

shell> 
```

Now we can access the `get_paraboloid_comp` function from Python, and thus grab the OpenMDAOCore.jl component.
In `example_python_package_openmdao_jl/scripts` there is a run script called `run_paraboloid.py` that shows how it's used.
The beginning of that file looks like this:

```
import juliacall
import openmdao.api as om
from omjlcomps import JuliaExplicitComp
from example_python_package_openmdao_jl.paraboloid import get_parabaloid_comp

prob = om.Problem()

jlcomp = get_parabaloid_comp()
parab_comp = JuliaExplicitComp(jlcomp=jlcomp)
prob.model.add_subsystem("parab_comp", parab_comp)
```

There we have used `get_paraboloid_comp` to get the paraboloid `struct`, passed it to `JuliaExplicitComp` to create an actual OpenMDAO component, and then added that to the OpenMDAO model.


## Fourth tip: use the `Manifest.in` file to include non-Python files in a Python package
When creating a source distribution for a Python package, `setuptools` will only include [certain files](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html), and, perhaps not surprisingly, will ignore Julia files.
The solution is to add a `MANIFEST.in` to the top-level of the Python package that includes all the Julia files you need, using `include` entries to add individual files, and `graft` to include everything under a directory.
For example, the `Manifest.in` file for `example_python_package_openmdao_jl` looks like this:

```
shell> cat example_python_package_openmdao_jl/MANIFEST.in
graft ./example_python_package_openmdao_jl/julia/JuliaParaboloidComponent
include ./example_python_package_openmdao_jl/paraboloid.jl
include ./example_python_package_openmdao_jl/circle.jl
include ./example_python_package_openmdao_jl/juliapkg.json

shell> 
```
