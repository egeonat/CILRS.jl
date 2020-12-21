using Plots
using AutoGrad
using Knet
include("carla100_dataset.jl")
include("cilrs_model.jl")


function train(epochs)
    model = CILRSModel(dropout_ratio=0.0)
    dataset = read_dataset("/home/onat/carla/cilrs-julia/datasets/tiny_CARLA100/", batchsize=1)
    optimizer = Adam(lr=0.00002)
    
    train_loss = zeros(epochs + 1)
    train_loss[1] = model(dataset)
    println("Epoch ", 0, " loss: ", train_loss[1])
    println("-"^80)
    for i in 2:epochs+1
        Knet.minimize!(model, dataset, optimizer)
        println("-"^80)
        train_loss[i] = model(dataset)
        println("Epoch ", i-1, " loss: ", train_loss[i])
        println("-"^80)
    end
    return train_loss
end

num_epochs = 10
train_loss = train(num_epochs)
plot(0:num_epochs, train_loss)