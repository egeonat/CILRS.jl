using Plots
using AutoGrad
using Knet
include("carla100_dataset.jl")
include("cilrs_model.jl")


function train(epochs)
    model = CILRSModel(dropout_ratio=0.0, pretrained=false)
    for p in params(model)
        p.opt = Adam(lr=0.0002)
    end
    dataset = read_dataset("/datasets/CARLA100", batchsize=120)
    
    train_loss = zeros(epochs)
    for i in 1:epochs
        println("Epoch ", i)
        # This is done without minimize function for easier debugging
        train_loss[i] = 0.0
        batch_count = 0
        for (x, y) in dataset
            loss = @diff model(x, y)
            train_loss[i] += value(loss)
            batch_count += 1
            for p in params(model)
                g = grad(loss, p)
                update!(value(p), g, p.opt)
            end
        end
        train_loss[i] /= batch_count
        println("Train epoch loss: ", train_loss[i])
    end
    return train_loss
end

num_epochs = 20
train_loss = train(num_epochs)
plot(1:num_epochs, train_loss[1:end], ylim=(0,2), yticks=0:0.2:2)
