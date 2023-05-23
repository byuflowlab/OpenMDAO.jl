module OpenMDAODocs

using Documenter, OpenMDAOCore, OpenMDAO

function main()
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
                      "API Reference"=>"reference.md",
                      "Limitations"=>"limitations.md"])
    if get(ENV, "CI", nothing) == "true"
        deploydocs(repo="github.com/byuflowlab/OpenMDAO.jl.git", devbranch="master")
    end
end

if !isinteractive()
    main()
end

end # module
