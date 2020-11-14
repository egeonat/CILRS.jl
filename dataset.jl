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
        img = PermutedDimsArray(channelview(float32.(img)), (2, 3, 1))
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

""" 
    read_dataset(root_dir, batch_size=1)

Read dataset returns an iterator of minibatches. Each minibatch is a tuple of
(input_array, output_array). 

An input_array is an array of size=batch_size containing tuples of input data. 
An output_array is an array of size=batch_size containing tuples of output data.    

Input tuples are of the form (rgb_img, speed, command) 
Action tuples are of the form (throttle, steer)
"""
(input, action)
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
    input_list = Array{Any}(undef, 0, 3)
    action_list = Array{Any}(undef, 0, 2)
    for route in routes
        rgb_data = dataset[route]["rgb"]
        measurements = dataset[route]["measurements"]
        speed_data = map(x -> x["speed"], measurements)
        command_data = map(x -> x["command"], measurements)

        throttle_data = map(x -> x["throttle"], measurements)
        steer_data = map(x -> x["steer"], measurements)
        
        inputs = hcat(rgb_data, speed_data, command_data)
        input_list = vcat(input_list, inputs)

        actions = hcat(throttle_data, steer_data)
        action_list = vcat(action_list, actions)
    end
    input_list = permutedims(input_list, (2, 1))
    action_list = permutedims(action_list, (2, 1))
    return (input_list, action_list)
end
