include("resnet34.jl")
include("loss.jl")

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
    action = SequentialModule([
        DenseLayer(512, 512, fn=relu),
        DenseLayer(512, 2, fn=tanh)
    ])
    CILRSModel(perception, measurements, fused_process, speed_pred, action)
end

function (m::CILRSModel)(rgb, speed, command)
    rgb_fts = m.perception(rgb)
    #println("rgb_fts: ", mean(rgb_fts))
    #println("rgb_fts shape: ", size(rgb_fts))
    spd_fts = m.measurements(reshape(speed, (1, size(speed)...)))
    #println("spd_fts: ", mean(spd_fts))
    #println("spd_fts shape: ", size(spd_fts))
    spd_prd = m.speed_pred(rgb_fts)
    fused_fts = m.fused_process(vcat(rgb_fts, spd_fts))
    #println("fused_fts: ", mean(fused_fts))
    action_prd = m.action(fused_fts)
    preds = Dict(
        "throttle"=>action_prd[1,:],
        "steer"=>action_prd[2,:],
        "speed"=>spd_prd
    )
end
function (m::CILRSModel)(x, y)
    rgb, speed, command = x
    #println("RGB size: ", size(rgb))
    #println("RGB mean: ", mean(rgb))
    #println("Speed size", size(speed))
    #println("Speed: ", speed)
    
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