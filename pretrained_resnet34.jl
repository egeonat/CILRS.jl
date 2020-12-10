using PyCall
include("resnet34.jl")
@pyimport torch
@pyimport numpy

function load_conv_weights!(conv_layer::ConvLayer, py_conv_layer)
    conv_layer.w[:] = param(permutedims(py_conv_layer.weight.data.numpy(), [3, 4, 2, 1]))[:]
    if !isnothing(py_conv_layer.bias)
        conv_layer.b[:] = param(test_conv2d.bias.data.numpy())
    end
    conv_layer
end

function load_basic_block_weights!(basic_block::BasicBlock, py_basic_block)
    bb_layers = basic_block.sequential_module.layers
    load_conv_weights!(bb_layers[1], py_basic_block.conv1)
    load_conv_weights!(bb_layers[4], py_basic_block.conv2)
    if typeof(basic_block.downsample) <: SequentialModule
        load_conv_weights!(basic_block.downsample.layers[1], py_basic_block.downsample["0"])
    end
    basic_block
end

function load_res_layer_weights!(res_layer::SequentialModule, py_res_layer)
    for (i, bb) in enumerate(res_layer.layers)
        i -= 1
        load_basic_block_weights!(bb, py_res_layer["$i"])
    end
end

# Loads weights of all layers except for fully connected layers
function load_ResNet_weights!(resnet::ResNet, torch_model)
    load_conv_weights!(resnet.sequential_module.layers[1], torch_model.conv1)
    load_layer_weights!(resnet.sequential_module.layers[5], torch_model.layer1)
    load_layer_weights!(resnet.sequential_module.layers[6], torch_model.layer2)
    load_layer_weights!(resnet.sequential_module.layers[7], torch_model.layer3)
    load_layer_weights!(resnet.sequential_module.layers[8], torch_model.layer4)
end

function ResNet34(pretrained::Bool=true, pretrained_path::String="resnet/resnet_34_model")
    model = ResNet([3, 4, 6, 3], 512)
    if pretrained
        pretrained_model = torch.load()
        load_ResNet_weights!(model, pretrained_model)
    end
    model
end