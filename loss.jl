using Statistics

function loss(preds, gt)
    throttle_error = mean(abs.(preds["throttle"] .- gt["throttle"]))
    steer_error = mean(abs.(preds["steer"] .- gt["steer"]))
    speed_error = mean(abs.(preds["speed"] .- gt["speed"]))

    println("Throttle preds: ", preds["throttle"][1])
    println("Throttle gt: ", gt["throttle"][1])
    println("Steer preds: ", preds["steer"][1])
    println("Steer gt: ", gt["steer"][1])
    # TODO add speed error back
    loss = throttle_error*0.5 + steer_error*0.45 
end