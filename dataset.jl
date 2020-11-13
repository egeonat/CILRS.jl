using JSON
using ImageIO
using Images
using Knet

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

function read_dataset(root_dir, batch_size=1)
    dataset = Dict()
    routes = readdir(root_dir)
    counter = 1
    for route in routes
        println("Reading route: ", counter, " out of ", size(routes)[end])
        counter += 1
        route_full_path = joinpath(root_dir, route)

        sample_data = Dict()
        sample_data["measurements"] = read_json(joinpath(route_full_path, "measurements"))
        sample_data["rgb"] = read_rgb(joinpath(route_full_path, "rgb"))

        dataset[route] = sample_data
    end
    dataset = format_dataset(dataset, routes)
    dataset = minibatch(dataset[1], dataset[2], batch_size, shuffle=true)
end

function format_dataset(dataset, routes)
    input_list = []
    action_list = []
    for route in routes
        rgb_data = dataset[route]["rgb"]
        measurements = dataset[route]["measurements"]
        speed_data = map(x -> x["speed"], measurements)
        command_data = map(x -> x["command"], measurements)

        throttle_data = map(x -> x["throttle"], measurements)
        steer_data = map(x -> x["steer"], measurements)

        input = hcat(rgb_data, speed_data, command_data)
        input = mapslices(x -> tuple(x...), input, dims=2)
        append!(input_list, input)

        actions = hcat(throttle_data)
        actions = mapslices(x -> tuple(x...), actions, dims=2)
        append!(action_list, actions)
    end
    return (input_list, action_list)
end
