using Statistics

function loss(preds, gt)
    throttle_error = mean(abs.(preds["throttle"] .- gt["throttle"]))
    steer_error = mean(abs.(preds["steer"] .- gt["steer"]))
    speed_error = mean(abs.(preds["speed"] .- gt["speed"]))

    #println("Throttle preds: ", preds["throttle"][1])
    #println("Throttle gt: ", gt["throttle"][1])
    #println("Steer preds: ", preds["steer"][1])
    #println("Steer gt: ", gt["steer"][1])
    # TODO add speed error back
    loss = throttle_error*0.5 + steer_error*0.45 + speed_error*0.05
end

# function loss(preds, gt)
#     throttle_error = mean(abs.(preds[1] .- gt[1]))
#     steer_error = mean(abs.(preds[2] .- gt[2]))
#     speed_error = mean(abs.(preds[3] .- gt[3]))

#     # TODO add speed error back
#     loss = throttle_error*0.5 + steer_error*0.5 
# end