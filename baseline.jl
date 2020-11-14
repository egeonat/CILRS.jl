using Knet, Statistics, Random


function init_model()
    w = Any[
        randn(Float32, (3, 3, 3, 8)) .* 0.1,
        randn(Float32, (3, 3, 8, 16)) .* 0.1,
        randn(Float32, (20, 64 * 36 * 16 + 1)) .* 0.1,
        zeros((20)),
        randn(Float32, (2, 20)) .* 0.1,
        zeros((2))
        ]
    w = map(Knet.array_type[], w)    
end

function conv_layer(w, x)
    x = conv4(w[1], x, padding=1, stride=2)
    x = relu.(x)
end

function lin_layer(w, x)
    x = w[1] * x
    x = w[2] .+ x
end

function predict(w, x, speed)
    x = conv_layer(w[1:1], x)
    x = conv_layer(w[2:2], x)
    x = lin_layer(w[3:4], vcat(mat(x), transpose(speed)))
    x = lin_layer(w[5:6], x)
end

function loss(w, x, speed, actions)
    act_preds = predict(w, x, speed)
    sum(abs.(act_preds .- actions))
end

struct Baseline
    w
end
Baseline() = Baseline(init_model())
(b::Baseline)(rgb, speed) = (predict(b.w, rgb, speed))
(b::Baseline)(rgb, speed, actions) = loss(b.w, rgb, speed, actions)