module OpenMDAODocs

using Documenter, OpenMDAOCore

function main()
    makedocs(
             sitename="OpenMDAO.jl",
             modules = [OpenMDAOCore],
             format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
             pages = ["Home" => "index.md", "The Python-Centric Approach" => "python_centric.md"])
end

end
