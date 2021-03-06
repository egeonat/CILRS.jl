using Statistics

function loss(preds, gt)
    #println(summary(preds["throttle"]))
    #println(summary(gt["throttle"]))
    #println(summary(preds["steer"]))
    #println(summary(gt["steer"]))
    #println(summary(preds["speed"]))
    #println(summary(gt["speed"]))

    speed_error = mean(abs.(preds["speed"] .- gt["speed"]))
    throttle_error = mean(abs.(preds["throttle"] .- gt["throttle"]))
    steer_error = mean(abs.(preds["steer"] .- gt["steer"]))

    #println("throttle preds: ", preds["throttle"][1])
    #println("throttle gt: ", gt["throttle"][1])
    #println("throttle loss: ", throttle_error)
    #println("steer preds: ", preds["steer"][1])
    #println("steer gt: ", gt["steer"][1])
    #println("steer loss: ", steer_error)
    #println("speed preds: ", preds["speed"][1])
    #println("speed gt: ", gt["speed"][1])
    #println("speed loss: ", speed_error)

    loss = throttle_error*0.5 + steer_error*0.45 + speed_error*0.05
end
