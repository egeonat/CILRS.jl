using Knet

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
(n::SequentialModule)(x) = (for l in n.layers;x = l(x); end; x;)

struct DenseLayer
    W
    b
    fn
end
function DenseLayer(in_size::Int, out_size::Int; fn::Function=identity)
    W = param(out_size, in_size, init=kaiming_normal)
    b = param(out_size, 1, init=zeros)
    DenseLayer(W, b, fn)
end
(l::DenseLayer)(x) = l.fn.(l.W*x .+ l.b)
    
struct ConvLayer
    W
    b
    pad
    stride
    fn
end
function ConvLayer(size::Int, in_ch::Int, out_ch::Int; pad::Int, stride::Int,
    fn::Function=identity, bias::Bool=true)
    W = param(size, size, in_ch, out_ch, init=kaiming_normal)
    b = nothing
    if bias
        b = param(1, 1, out_ch, init=zeros)
    end
    ConvLayer(W, b, pad, stride, fn)
end
function (c::ConvLayer)(x) 
    out = conv4(c.W, x, padding=c.pad, stride=c.stride)
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
BNormLayer2d(channels::Int) = BNormLayer2d(bnmoments(), bnparams(channels))
(bn::BNormLayer2d)(x) = batchnorm(x, bn.bmoments, bn.bparams)

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

function make_layer(in_ch::Int, out_ch::Int, blocks::Int; stride::Int=1)
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

struct ResNet
    sequential_module
end
function ResNet(layers::Array{Int, 1}, out_size::Int)
    sequential_module = SequentialModule([
        ConvLayer(7, 3, 64, stride=2, pad=3, bias=false),
        BNormLayer2d(64),
        x -> relu.(x),
        PoolLayer(3, stride=2, pad=1),
        make_layer(64, 64, layers[1]),
        make_layer(64, 128, layers[2], stride=2),
        make_layer(128, 256, layers[3], stride=2),
        make_layer(256, 512, layers[4], stride=2),
        x -> global_avg_pool2d(x),
        x -> mat(x),
        DenseLayer(512, out_size)
    ])
    ResNet(sequential_module)
end
(r::ResNet)(x) = r.sequential_module(x)