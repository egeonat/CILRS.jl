include("carla100_dataset.jl")

dataset = read_sections(["/datasets/CARLA100/CVPR2019-CARLA100_03"]; batchsize=120)
save("/kuacc/users/eozsuer16/dl/CILRS.jl/preloaded_datasets/CARLA100_03.jld", "dataset", dataset)
