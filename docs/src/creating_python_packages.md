```@meta
CurrentModule = OpenMDAODocs
```
# Creating Python Packages That Depend On OpenMDAOCore.jl
The OpenMDAO.jl repository contains an example Python package that implements a couple of the examples described in the previous docs (the [`SparseADExplicitComp` Paraboloid](@ref) and [`SparseADExplicitComp` with Actual Sparsity](@ref) examples) in the `examples/example_python_package_openmdao_jl/` sub-directory.
The package structure looks like this:

```
├── example_python_package_openmdao_jl
    ├── MANIFEST.in
    ├── example_python_package_openmdao_jl
    │   ├── __init__.py
    │   ├── circle.jl
    │   ├── circle.py
    │   ├── julia
    │   │   ├── JuliaExampleComponents
    │   │       ├── Project.toml
    │   │       ├── src
    │   │           ├── JuliaExampleComponents.jl
    │   │           ├── circle.jl
    │   │           └── paraboloid.jl
    │   ├── juliapkg.json
    │   ├── paraboloid.jl
    │   ├── paraboloid.py
    │   ├── test
    │       ├── test_circle_example.py
    │       └── test_paraboloid_example.py
    ├── pyproject.toml
    ├── scripts
        ├── run_circle.py
        └── run_paraboloid.py
```
You can look through that to see how things are put together, but I'll point out a few "best-practices" that I've found.

## First step: create a python package that depends on `omjlcomps`
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

It also depends on `openmdao` (of course), and [`juliapkg`](https://github.com/JuliaPy/pyjuliapkg) and [`juliacall`](https://github.com/JuliaPy/PythonCall.jl).
`juliapkg` is a Python package that allows us to define Julia dependencies of Python packages, and `juliacall` is what we use to call Julia code from python.
But how will we, specifically, declare which Julia packages we need?
That's what the `juliapkg.json` file is for, which `juliapkg` reads when deciding what Julia packages are needed for a given Python package.
After reading the [`juliapkg` docs on the subject](https://github.com/JuliaPy/pyjuliapkg?tab=readme-ov-file#declare-dependencies), we could edit that file directly, listing out each Julia package that we need in excruciating detail.
Or we could use `juliapkg`'s functional interface, running commands like `juliapkg.add` from a Python REPL.
I find that kind of tedious, and instead like to have a short `juliapkg.json` file that looks like this:

```
shell> cat example_python_package_openmdao_jl/example_python_package_openmdao_jl/juliapkg.json
{"packages": {
    "JuliaExampleComponents": {"uuid": "4f289198-a466-4861-a6bd-1e7b09ed8707", "dev": true, "path": "./julia/JuliaExampleComponents"}
    }
}

shell> 
```

That says that our example Python package will depend on just one Julia package named `JuliaExampleComponents`, which is located under `julia/JuliaExampleComponents` (relative to the location of the `juliapkg.json` file itself).


## Create a Julia package within the Python source code to easily track Julia dependencies
In the directory structure above, there is a `JuliaExampleComponents` directory that is a plain old [Julia package](https://pkgdocs.julialang.org/v1/creating-packages/).

## Use the `Manifest.in` file to include non-Python files in a Python package
When creating a source distribution for a Python package, `setuptools` will only include [certain files](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html), and, perhaps not surprisingly, will ignore Julia files.
The solution is to add a `MANIFEST.in` to the top-level of the Python package that includes all the Julia files you need.
In the 


