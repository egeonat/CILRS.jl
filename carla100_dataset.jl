import Base: length, iterate
using Random
using JSON
using ImageIO
using Images
using Knet
using JLD

# TODO recompute these with larger sample
RGB_MEAN = [0.28462812, 0.3055363, 0.32060984]
RGB_STD_DEV = [0.2486219, 0.24289215, 0.23690315]

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
    x = (Knet.atype()(d.rgb[:,:,:,state[1:d.batchsize]]),
        Knet.atype()(d.speed[state[1:d.batchsize]]),
        Knet.atype()(d.command[state[1:d.batchsize]]))
    y = (Knet.atype()(d.throttle[state[1:d.batchsize]]),
        Knet.atype()(d.steer[state[1:d.batchsize]]))
    sample = (x, y)
        
    state = state[d.batchsize+1:end]
    return (sample, state)
end

function length(d::Carla100Data)
    return length(d.command)
end

function read_dataset(root_dir; batchsize=1, section_lim=-1, episode_lim=-1, shuffle=true)
    section_dirs = [s for s in readdir(root_dir, join=true) if occursin("CVPR2019-CARLA100_", s)]
	if section_lim != -1
		section_dirs = section_dirs[1:section_lim]
	end
    read_sections(section_dirs, batchsize=batchsize, episode_lim=episode_lim)
end

function read_sections(section_dirs; batchsize=1, episode_lim=-1, shuffle=true)
    rgb = Array{Float32, 4}(undef, 88, 200, 3,0)
    speed = Array{Float32}(undef, 0)
    command = Array{Int8}(undef, 0)
    throttle = Array{Float32}(undef, 0)
    steer = Array{Float32}(undef, 0)

	println("Reading dataset with ", Threads.nthreads(), " threads.")
	for section in section_dirs
        println("Reading section: ", section)
        episodes = readdir(section, join=true)
		if episode_lim != -1
			episodes = episodes[1:episode_lim]
		end

        ep_rgb = Array{Array{Float32, 4}}(undef, length(episodes))
        ep_speed = Array{Array{Float32}}(undef, length(episodes))
        ep_command = Array{Array{Int8}}(undef, length(episodes))
        ep_throttle = Array{Array{Float32}}(undef, length(episodes))
        ep_steer = Array{Array{Float32}}(undef, length(episodes))

        Threads.@threads for i in 1:length(episodes) 
            ep_rgb[i], ep_speed[i], ep_command[i], ep_throttle[i], ep_steer[i] = read_episode(episodes[i])
        end
        rgb = cat(rgb, ep_rgb..., dims=4)
        speed = cat(speed, ep_speed..., dims=1)
        command = cat(command, ep_command..., dims=1)
        throttle = cat(throttle, ep_throttle..., dims=1)
        steer = cat(steer, ep_steer..., dims=1)
    end
    println("Loaded ", length(speed), " samples")
    #println(summary(rgb))
    #println(summary(speed))
    #println(summary(command))
    #println(summary(throttle))
    #println(summary(steer))
    Carla100Data(rgb, speed, command, throttle, steer, batchsize, shuffle)
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
    if length(rgb_files) != length(json_files)
		println("Rgb and json file count mismatch")
	end
	flush(stdout)
	num_samples = min(length(rgb_files ), length(json_files ))
    for i in 0:num_samples-1
		id_str = string(lpad(i, 5, "0"))
        # Load json measurements
        j_dict = JSON.parsefile(joinpath(episode, string("measurements_", id_str, ".json")))
		# Skip if keys are missing
		if !haskey(j_dict, "playerMeasurements") ||
			!haskey(j_dict["playerMeasurements"], "forwardSpeed") ||
			!haskey(j_dict, "directions") ||
			!haskey(j_dict, "throttle") ||
			!haskey(j_dict, "brake") ||
			!haskey(j_dict, "steer")
			#println("Missing keys in ", episode, " id ", id_str, ". Skipping")
			#println("measurements_", id_str, ".json keys: ")
			#println.(keys(j_dict))
			continue
		end
        # Load and reshape rgb
        rgb = float32.(load(joinpath(episode, string("CentralRGB_", id_str, ".png"))))
        rgb = PermutedDimsArray(channelview(rgb), (2, 3, 1))
		rgb = reshape(rgb, size(rgb)..., 1)
        # Normalize
        #rgb = (rgb .- reshape(RGB_MEAN, (1, 1, 3, 1))) 
        #rgb = rgb ./ reshape(RGB_STD_DEV, (1, 1, 3, 1))
        # Extract relevant measurements
        speed = j_dict["playerMeasurements"]["forwardSpeed"]
        command = j_dict["directions"]
        throttle = nothing
        if j_dict["throttle"] != 0.0
            throttle = j_dict["throttle"]
        else
            throttle = j_dict["brake"] * (-1)
        end
        steer = j_dict["steer"]

        ep_rgb = cat(ep_rgb, rgb, dims=4)
        push!(ep_speed, speed)
        push!(ep_command, command)
        push!(ep_throttle, throttle)
        push!(ep_steer, steer)
    end
	println("Finished reading ", episode)
    ep_rgb, ep_speed, ep_command, ep_throttle, ep_steer
end

function read_preloaded_dataset(path)
	load(path)["dataset"]
end

dataset = read_sections(["/datasets/CARLA100/CVPR2019-CARLA100_02"]; batchsize=120, episode_lim=100)
save("/kuacc/users/eozsuer16/dl/CILRS.jl/preloaded_datasets/CARLA_100_ep300-400.jld_bs120.jld", "dataset", dataset)
