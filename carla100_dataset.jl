import Base: length, iterate
using Random
using JSON
using ImageIO
using Images
using Knet

struct Carla100Data
    rgb::Array{Float32, 4}
    speed::Array{Float32}
    command::Array{Int8}
    throttle::Array{Float32}
    steer::Array{Float32}

    batchsize::Int
    shuffle::Bool
end

function iterate(d::Carla100Data, state=ifelse(d.shuffle, randperm(length(d)), collect(1:length(d))))
    remaining_count = length(state)
    if remaining_count < d.batchsize
        return nothing
    end
    sample = (KnetArray{Float32, 4}(d.rgb[:,:,:,state[1:d.batchsize]]),
        KnetArray{Float32}(d.speed[state[1:d.batchsize]]),
        KnetArray{Int8}(d.command[state[1:d.batchsize]]),
        KnetArray{Float32}(d.throttle[state[1:d.batchsize]]),
        KnetArray{Float32}(d.steer[state[1:d.batchsize]]))
    state = state[d.batchsize+1:end]
    println(length(state))
    return (sample, state)
end

function length(d::Carla100Data)
    return length(d.command)
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
function read_dataset(root_dir; batch_size=1)
    section_dirs = [s for s in readdir(root_dir, join=true) if occursin("CVPR2019-CARLA100_", s)]
    read_sections(section_dirs, batch_size=batch_size)
end

function read_sections(section_dirs; batch_size=1, shuffle=true)
    rgb = Array{Float32, 4}(undef, 88, 200, 3,0)
    speed = Array{Float32}(undef, 0)
    command = Array{Int8}(undef, 0)
    throttle = Array{Float32}(undef, 0)
    steer = Array{Float32}(undef, 0)

    for section in section_dirs
        println("Reading section: ", section)
        episodes = readdir(section, join=true)

        ep_rgb = Array{Array{Float32, 4}}(undef, length(episodes))
        ep_speed = Array{Array{Float32}}(undef, length(episodes))
        ep_command = Array{Array{Int8}}(undef, length(episodes))
        ep_throttle = Array{Array{Float32}}(undef, length(episodes))
        ep_steer = Array{Array{Float32}}(undef, length(episodes))

        for i in 1:length(episodes)
            ep_rgb[i], ep_speed[i], ep_command[i], ep_throttle[i], ep_steer[i] = read_episode(episodes[i])
        end
        rgb = cat(rgb, ep_rgb..., dims=4)
        speed = cat(speed, ep_speed..., dims=1)
        command = cat(command, ep_command..., dims=1)
        throttle = cat(throttle, ep_throttle..., dims=1)
        steer = cat(steer, ep_steer..., dims=1)
    end
    Carla100Data(rgb, speed, command, throttle, steer, batch_size, shuffle)
end

function read_episode(episode)
    println("Reading episode: ", episode)
    ep_files = readdir(episode, join=true)
    ep_rgb = Array{Float32, 4}(undef, 88, 200, 3, 0)
    ep_speed = Array{Float32}(undef, 0)
    ep_command = Array{Int8}(undef, 0)
    ep_throttle = Array{Float32}(undef, 0)
    ep_steer = Array{Float32}(undef, 0)

    rgb_files = filter(x -> occursin("CentralRGB", x), ep_files)
    json_files = filter(x -> occursin("measurements", x), ep_files)
    @assert length(rgb_files) == length(json_files) "Rgb and json file count mismatch"
    for i in 0:length(rgb_files)-1
        id_str = string(lpad(i, 5, "0"))
        j_dict = JSON.parsefile(joinpath(episode, string("measurements_", id_str, ".json")))
        rgb = float32.(load(joinpath(episode, string("CentralRGB_", id_str, ".png"))))
        rgb = PermutedDimsArray(channelview(rgb), (2, 3, 1))
        rgb = reshape(rgb, size(rgb)..., 1)
        speed = j_dict["playerMeasurements"]["forwardSpeed"]
        command = j_dict["directions"]
        throttle = nothing
        if j_dict["throttle"] == 0.0
            throttle = j_dict["throttle"]
        elseif j_dict["brake"] == 0.0
            throttle = j_dict["brake"] * (-1)
        else
            @assert false, println("Throttle: ", j_dict["throttle"], " - Speed: ", j_dict["speed"])
        end
        steer = j_dict["steer"]

        ep_rgb = cat(ep_rgb, rgb, dims=4)
        push!(ep_speed, speed)
        push!(ep_command, command)
        push!(ep_throttle, throttle)
        push!(ep_steer, steer)
    end
    ep_rgb, ep_speed, ep_command, ep_throttle, ep_steer
end