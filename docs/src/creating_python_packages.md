```@meta
CurrentModule = OpenMDAODocs
```
# Creating Python Packages That Depend On OpenMDAOCore.jl
The OpenMDAO.jl repository contains an example Python package that implements a couple of the examples described in the previous docs (the [`SparseADExplicitComp` Paraboloid](@ref) and [`SparseADExplicitComp` with Actual Sparsity](@ref) examples) in the `examples/example_python_package_openmdao_jl/` sub-directory.
You can look through that to see how things are put together, but I'll point out a few "best-practices" that I've found.

## Remember to use the `Manifest.in` file to include non-Python files in a Python package
When creating a source distribution for a Python package, `setuptools` will only include [certain files](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html), and, perhaps not surprisingly, will ignore Julia files.
The solution is to add a `MANIFEST.in` to the top-level of the Python package that includes all the Julia files you need.

## Create a Julia package within the Python source code to track dependencies

