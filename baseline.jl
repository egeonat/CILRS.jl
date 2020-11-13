using Knet, Statistics


function init_model()
    w = Any[
        randn(Float32, (3, 3, 3, 8)) .* 0.1,
        randn(Float32, (3, 3, 8, 16)) .* 0.1,
        randn(Float32, (20, 64 * 36 * 16)) .* 0.1,
        randn(Float32, (2, 20)) .* 0.1
        ]
    w = map(Knet.array_type[], w)    

function conv_layer(w, x)
    x = conv4(w[1], x, padding=1, stride=2)
    x = relu.(x)

function lin_layer(w, x)
    x = w[1] * x
    x = w[2] .+ x

function predict(w, x)
    x = conv_layer(w[1:1], x)
    x = conv_layer(w[2:2], x)
    x = lin_layer(w[3:4], mat(x))
    x = lin_layer(w[5:6], x)


function loss(w, x, actions)