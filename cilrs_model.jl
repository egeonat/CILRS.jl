include("resnet34.jl")
include("loss.jl")
include("utils.jl")

struct BranchedModule
    branches
end
function (m::BranchedModule)(x, commands)
    all_outputs = []
    for (i, b) in enumerate(m.branches)
        push!(all_outputs, b(x))
    end
    #println(all_outputs)
    #println(commands)
    out_list = all_outputs[Int(commands[1])][:,1]
    for i in 2:size(x)[end]
        index = Int(commands[i])
        out = all_outputs[index][:,i]
        out_list = hcat(out_list, out)
    end
    #println(out_list)
    out_list
end
        
struct CILRSModel
    perception
    measurements
    fused_process
    speed_pred
    action
end

function CILRSModel(;pretrained, dropout_ratio=0.0)
    perception = ResNet34(pretrained=pretrained)
    measurements = SequentialModule([
        DenseLayer(1, 128, fn=relu),
        DenseLayer(128, 128, fn=relu)
    ])
    fused_process = DenseLayer(640, 512, fn=relu)
    speed_pred = SequentialModule([
        DenseLayer(512, 256, fn=relu),
        DenseLayer(256, 256),
        x -> dropout(x, dropout_ratio),
        x -> relu.(x),
        DenseLayer(256, 1)
    ])
    branches = Array{SequentialModule}(undef, 0)
    for i in 1:4
        branch = SequentialModule([
            DenseLayer(512, 512, fn=relu),
            DenseLayer(512, 2, fn=identity)
            ])
        push!(branches, branch)
    end
    action = BranchedModule(branches)
    CILRSModel(perception, measurements, fused_process, speed_pred, action)
end

function (m::CILRSModel)(rgb, speed, command)
    # Visualize first rgb image
    #visualize_rgb(rgb)

    rgb_fts = m.perception(rgb)
    #rgb_fts = Knet.atype()(ones(512, 120))

    spd_fts = m.measurements(reshape(speed, (1, size(speed)...)))
    #spd_fts = Knet.atype()(ones(128))

    spd_prd = m.speed_pred(rgb_fts)
    fused_fts = m.fused_process(vcat(rgb_fts, spd_fts))

    action_prd = m.action(fused_fts, command)
    preds = Dict(
        "throttle"=>action_prd[1,:],
        "steer"=>action_prd[2,:],
        "speed"=>spd_prd
    )
end

function (m::CILRSModel)(x, y)
    rgb, speed, command = x
    
    throttle, steer = y
    preds = m(rgb, speed, command)
    gt = Dict(
        "throttle"=>throttle,
        "steer"=>steer,
        "speed"=>speed
    )
    loss(preds, gt)
end
function (m::CILRSModel)(dataset)
    loss_sum = 0.0
    count = length(dataset)
    for (x, y) in dataset
        loss_sum += m(x, y)
    end
    loss_sum / count
end
