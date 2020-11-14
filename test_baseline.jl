include("dataset.jl")
include("baseline.jl")

batches = read_dataset("/home/onat/carla/data/sample_dataset/", 2)
inputs, actions = first(batches)

rgb = KnetArray{Float32, 4}(cat(inputs[1, :]..., dims=(4)))
speed = KnetArray{Float32}(inputs[2, :])
command = KnetArray{Float32}(inputs[3, :])

actions = KnetArray{Float32, 2}(actions[1:2, :])

model = Baseline()

println("Predictions:")
println(model(rgb, speed))
println("Loss")
println(model(rgb, speed, actions))