using Statistics

function loss(preds, gt)
    throttle_error = mean(abs.(preds["throttle"] .- gt["throttle"]))
    steer_error = mean(abs.(preds["steer"] .- gt["steer"]))
    speed_error = mean(abs.(preds["speed"] .- gt["speed"]))

    #println("Preds: ", preds["throttle"][1])
    #println("Gt: ", gt["throttle"][1])
    loss = throttle_error*0.5 + steer_error*0.45 + speed_error*0.05
end