module OpenMDAODocs

using Documenter, OpenMDAOCore, OpenMDAO

function doit()
    makedocs(
             sitename="OpenMDAO.jl",
             modules = [OpenMDAOCore, OpenMDAO],
             format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
             doctest=true,
             pages = ["Home"=>"index.md",
                      "A Simple Example"=>"simple_paraboloid.md",
                      "A More Complicated Example"=>"nonlinear_circuit.md",
                      "Variable Shapes at Runtime"=>"shape_by_conn.md",
                      "A Simple Dymos Example"=>"brachistochrone.md",
                      "Auto-Sparse Examples"=>"auto_sparse_ad.md",
                      "Auto-Matrix-Free Examples"=>"matrix_free_ad.md",
                      "Creating Python Packages That Use OpenMDAOCore.jl"=>"creating_python_packages.md",
                      "API Reference"=>"reference.md",
                      "Limitations"=>"limitations.md",
                      "Developer Docs"=>"dev_docs.md"])
    if get(ENV, "CI", nothing) == "true"
        deploydocs(repo="github.com/byuflowlab/OpenMDAO.jl.git", devbranch="master")
    end
end

if !isinteractive()
    doit()
end

end # module
