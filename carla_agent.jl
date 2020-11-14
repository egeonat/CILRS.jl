include("baseline.jl")

model = Baseline()

function run_step(img, speed, command)
    img = convert.(Float32, img[:,:,1:3]) / 255
    img = reshape(img, (size(img)..., 1))
    img = KnetArray{Float32, 4}(img)

    speed = KnetArray{Float32}([speed])

    #println(typeof(img))
    #println(size(img))
    #println(typeof(speed))
    #println(command)

    preds = model(img, speed)

    #println(typeof(preds))
    #println(size(preds))
    throttle = preds[1,1]
    steer = preds[2,1]
    return return throttle, steer
end

