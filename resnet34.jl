using Knet
using PyCall
@pyimport torch
@pyimport numpy

function kaiming_normal(a...; mode="fan_out", gain=sqrt(2))
    w = randn(a...)
    if ndims(w) == 1
        fanout = 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanout = size(w, 1)
        fanin = size(w, 2)
    else
        fanin = div(length(w), a[end])
        fanout = div(length(w), a[end-1])
    end

    if mode == "fan_in"
        fan = fanin
    else
        fan = fanout
    end
    s = convert(eltype(w), sqrt(1 / fan)) * gain
    w .* s
end

struct SequentialModule
    layers
end
SequentialModule(layers...) = SequentialModule(layers)
(n::SequentialModule)(x) = (for l in n.layers;;x = l(x); end; x;)

struct DenseLayer
    w
    b
    fn
end
function DenseLayer(in_size::Int, out_size::Int; fn::Function=identity)
    w = param(out_size, in_size, init=xavier)
    b = param(out_size, 1, init=zeros)
    DenseLayer(w, b, fn)
end
function (l::DenseLayer)(x) 
    out = l.w*x
    out = out .+ l.b
    out = l.fn.(out)
    return out
end
    
struct ConvLayer
    w
    b
    pad
    stride
    fn
end
function ConvLayer(size::Int, in_ch::Int, out_ch::Int; pad::Int, stride::Int,
    fn::Function=identity, bias::Bool=true)
    w = param(size, size, in_ch, out_ch, init=xavier)
    b = nothing
    if bias
        b = param(1, 1, out_ch, init=zeros)
    end
    ConvLayer(w, b, pad, stride, fn)
end
function (c::ConvLayer)(x) 
    out = conv4(c.w, x, padding=c.pad, stride=c.stride)
    if !isnothing(c.b)
        out = out .+ c.b
    end
    c.fn.(out)
end

struct PoolLayer
    d
    stride
    pad
    mode
end
PoolLayer(d::Int; stride::Int=d, pad::Int=0, mode::Int=0) = PoolLayer(d, stride, pad, mode)
(p::PoolLayer)(x) = pool(x, window=p.d, stride=p.stride, padding=p.pad, mode=p.mode)

function global_avg_pool2d(x)
    pool(x, window=(size(x, 1), size(x, 2)))
end

struct BNormLayer2d
    bmoments
    bparams
end
BNormLayer2d(channels::Int) = BNormLayer2d(bnmoments(), Knet.atype(bnparams(channels)))
function (bn::BNormLayer2d)(x)
	#println("Bnorm input: ", summary(x))
	#println(summary(bn.bmoments))
	#println(summary(bn.bparams))
	batchnorm(x, bn.bmoments, bn.bparams)
end

struct BasicBlock
    sequential_module
    downsample
end
function BasicBlock(in_ch::Int, out_ch::Int; stride::Int=1, downsample=identity)
    sequential_module = SequentialModule([
    ConvLayer(3, in_ch, out_ch, pad=1, stride=stride, bias=false),
    BNormLayer2d(out_ch),
    x -> relu.(x),
    ConvLayer(3, out_ch, out_ch, pad=1, stride=1, bias=false),
    BNormLayer2d(out_ch),
    ])
    BasicBlock(sequential_module, downsample)
end
function (bb::BasicBlock)(x)
    residual = bb.downsample(x) 
    out = bb.sequential_module(x)
    relu.(out + residual)
end

function _make_layer(in_ch::Int, out_ch::Int, blocks::Int; stride::Int=1)
    downsample = identity
    if stride != 1 || in_ch != out_ch
        downsample = SequentialModule([
            ConvLayer(1, in_ch, out_ch, pad=0, stride=stride, bias=false),
            BNormLayer2d(out_ch)
        ])
    end
    layers = []
    push!(layers, BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample))
    for i in 2:blocks
        push!(layers, BasicBlock(out_ch, out_ch))
    end
    SequentialModule(layers)
end

# ResNet implementation support ResNet18 and Resnet34 only
function ResNet(layers::Array{Int, 1}, in_ch::Int, out_ch::Int)
    SequentialModule([
        ConvLayer(7, in_ch, 64, stride=2, pad=3, bias=false),
        BNormLayer2d(64),
        x -> relu.(x),
        PoolLayer(3, stride=2, pad=1),
        _make_layer(64, 64, layers[1]),
        _make_layer(64, 128, layers[2], stride=2),
        _make_layer(128, 256, layers[3], stride=2),
        _make_layer(256, 512, layers[4], stride=2),
        x -> global_avg_pool2d(x),
        x -> mat(x),
        DenseLayer(512, out_ch)
    ])
end

function _load_conv_weights!(conv_layer::ConvLayer, py_conv_layer)
    conv_layer.w[:] = param(permutedims(py_conv_layer.weight.data.numpy(), [3, 4, 2, 1]))[:]
    if !isnothing(py_conv_layer.bias)
        conv_layer.b[:] = param(test_conv2d.bias.data.numpy())
    end
    conv_layer
end

function _load_basic_block_weights!(basic_block::BasicBlock, py_basic_block)
    bb_layers = basic_block.sequential_module.layers
    _load_conv_weights!(bb_layers[1], py_basic_block.conv1)
    _load_conv_weights!(bb_layers[4], py_basic_block.conv2)
    if typeof(basic_block.downsample) <: SequentialModule
        _load_conv_weights!(basic_block.downsample.layers[1], py_basic_block.downsample["0"])
    end
    basic_block
end

function _load_layer_weights!(res_layer::SequentialModule, py_res_layer)
    for (i, bb) in enumerate(res_layer.layers)
        i -= 1
        _load_basic_block_weights!(bb, py_res_layer["$i"])
    end
end

# Loads weights of all convolutional layers except for fully connected layers
function _load_ResNet_weights!(resnet::SequentialModule, torch_model)
    _load_conv_weights!(resnet.layers[1], torch_model.conv1)
    _load_layer_weights!(resnet.layers[5], torch_model.layer1)
    _load_layer_weights!(resnet.layers[6], torch_model.layer2)
    _load_layer_weights!(resnet.layers[7], torch_model.layer3)
    _load_layer_weights!(resnet.layers[8], torch_model.layer4)
end

function ResNet34(;in_channels=3, out_channels=512, pretrained::Bool,
    pretrained_path::String="resnet/resnet_34_model")
    model = ResNet([3, 4, 6, 3], in_channels, out_channels)
    if pretrained
        println("Loading pretrained model from ", pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        _load_ResNet_weights!(model, pretrained_model)
    end
    model
end
