module OpenMDAODocs

using Documenter, OpenMDAOCore, OpenMDAO

function main()
    makedocs(
             sitename="OpenMDAO.jl",
             modules = [OpenMDAOCore, OpenMDAO],
             format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
             pages = ["Home"=>"index.md",
                      "A Simple Example"=>"simple_paraboloid.md",
                      "A More Complicated Example"=>"nonlinear_circuit.md",
                      "A Simple Dymos Example"=>"brachistochrone.md",
                      "API Reference"=>"reference.md",
                      "Limitations"=>"limitations.md"])
    if get(ENV, "CI", nothing) == "true"
        deploydocs(repo="github.com/dingraha/OpenMDAO.jl.git", devbranch="master")
    end
end

if !isinteractive()
    main()
end

end # module
