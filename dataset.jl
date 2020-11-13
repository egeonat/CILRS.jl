using JSON
using ImageIO
using Images

include("utils.jl")

function read_json(dir)
    json_list = []
    json_files = readdir(dir)
    for f in json_files
        f = joinpath(dir, f)
        
        #fix_json(f)

        j_dict = JSON.parsefile(f)
        push!(json_list, j_dict)
    end
    return json_list
end

function read_rgb(dir)
    rgb_list = []
    rgb_files = readdir(dir)
    for f in rgb_files
        img = load(joinpath(dir, f))
        img = channelview(img)
        push!(rgb_list, img)
    end
    return rgb_list
end
        

function read_samples(sample_dir)
    sample_data = Dict()
    sample_data["measurements"] = read_json(joinpath(sample_dir, "measurements"))
    sample_data["rgb"] = read_rgb(joinpath(sample_dir, "rgb"))
    return sample_data
end

function read_dataset(root_dir)
    dataset = Dict()
    route_names = readdir(root_dir)
    for route in route_names
        route_full_path = joinpath(root_dir, route)

        sample_data = Dict()
        sample_data["measurements"] = read_json(joinpath(route_full_path, "measurements"))
        sample_data["rgb"] = read_rgb(joinpath(route_full_path, "rgb"))

        dataset[route] = sample_data
    end
    return (dataset, route_names)
end

