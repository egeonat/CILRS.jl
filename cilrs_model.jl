include("resnet34.jl")

struct cilrs_model
    perception
    measurements
    fused_process
    speed_pred
    action
end
function cilrs_model()
    perception = ResNet34(pretrained=true)
    measurements = SequentialModule([
        DenseLayer(1, 128, fn=relu),
        DenseLayer(128, 128, fn=relu)
    ])
    fused_process = DenseLayer(640, 512, fn=relu)
    speed_pred = SequentialModule([
        DenseLayer(512, 256, fn=relu),
        DenseLayer(256, 256),
        x -> dropout(x, 0.5),
        x -> relu(x),
        DenseLayer(256, 1)
    ])
    action = SequentialModule([
        DenseLayer(512, 512, fn=relu),
        DenseLayer(512, 2, fn=tanh)
    ])
end