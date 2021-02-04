using AutoGrad
using Knet
using JLD2
include("carla100_dataset.jl")
include("cilrs_model.jl")
include("utils.jl")

function train(epochs, val_interval; dropout_ratio=0.5, learning_rate=0.0002)
    # Set up model
    # This should become pretrained=true after problems are fixed
    model = CILRSModel(dropout_ratio=dropout_ratio, pretrained=true)
    for p in params(model)
        p.opt = Adam(lr=learning_rate)
    end
    
    # Set up train and val data
    @load "/userfiles/eozsuer16/train.jld2" dataset
    train_set = dataset
    println("Train set of ", length(train_set), " samples and batchsize: ", train_set.batchsize)
    @load "/userfiles/eozsuer16/val.jld2" dataset
    val_set = dataset
    println("Val set of ", length(val_set), " samples and batchsize: ", val_set.batchsize)
    
    # Arrays to record loss values
    train_loss = zeros(epochs)
    val_loss = zeros(div(epochs, val_interval))

    # Training loop
    for i in 1:epochs
        println("Epoch ", i)
        batch_count = 0

        # Train
        for (x, y) in train_set
            iter_loss = @diff model(x, y)
            train_loss[i] += value(iter_loss)
            
            batch_count += 1
            for p in params(model)
                g = grad(iter_loss, p)
                #println(p)
                if !isnothing(g)
                    update!(value(p), g, p.opt)
                end
            end
        end
        train_loss[i] /= batch_count
        println("Train epoch loss: ", train_loss[i])

        # Val
        if i % val_interval == 0
            batch_count = 0
            for (x, y) in val_set
                iter_loss = model(x, y)
                val_loss[div(i, val_interval)] += value(iter_loss)
                batch_count += 1
            end
            val_loss[div(i, val_interval)] /= batch_count
            println("Val epoch loss: ", val_loss[div(i, val_interval)])
            println("Best val loss:  ",  minimum(val_loss[1:div(i, val_interval)]))
            if val_loss[div(i, val_interval)] <= minimum(val_loss[1:div(i, val_interval)])
                model_path = "models/" * string(i) * "_epochs_" * string(dropout_ratio) *
                    "dropout.jld2"
                println("Saving model: ", model_path)
                @save model_path model
            end
        end

        # Manually reduce learning rate like the original paper
        #if i % 20 == 0
        #   for p in params(model)
        #       p.opt.lr *= 0.5 
        #   end
        #end
        flush(stdout)
    end
    return train_loss, val_loss
end

Knet.setseed(10)
num_epochs = 200
dropout_ratio = 0.5
val_interval = 1
learning_rate = 0.0002
train(num_epochs, val_interval, dropout_ratio=dropout_ratio, learning_rate=learning_rate)
