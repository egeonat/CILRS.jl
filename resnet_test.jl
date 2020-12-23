# Script to test my resnet34 implementation on the mnist dataset
using CUDA
using Knet
using MLDatasets: MNIST
include("resnet34.jl")

function loss_nll(model, x, y)
	#println(summary(x))
	#println(summary(y))
    scores = model(x)
    loss = nll(scores, y)
    loss
end
function validate(model, dtst)
    acc = 0.0
    batch_count = 0
    for (x, y) in dtst
		#println(summary(x))
		#println(summary(y))
		# This is for testing ResNet with 3 input channels
		x = cat(x, x, x, dims=3)
        scores = model(x)
        acc += accuracy(scores, y)
        batch_count += 1
    end
    acc /= batch_count
end
function load_mnist_dataset(bsize)
	xtrn,ytrn = MNIST.traindata(Float32)
	xtst,ytst = MNIST.testdata(Float32)
	println("Loaded MNIST Data:")
	println.(summary.((xtrn,ytrn,xtst,ytst)))

	dtrn = minibatch(xtrn, (ytrn .+ 1), bsize, xsize=(size(xtrn)[1], size(xtrn)[2], 1, 100))
	dtst = minibatch(xtst, (ytst .+ 1), bsize, xsize=(size(xtrn)[1], size(xtrn)[2], 1, 100))
	println("Generated minibatches")
	println.(summary.((dtrn, ytrn)))
	dtrn, dtst
end

bsize = 100
dtrn, dtst = load_mnist_dataset(bsize)
model = ResNet34(pretrained=true, in_channels=3, out_channels=10)
for p in params(model)
	p.opt = Adam(lr=0.001)
end

println("Start accuracy :", validate(model, dtst))

epochs = 20
train_loss = zeros(epochs)
val_acc = zeros(epochs)
for i in 1:epochs
    println("Epoch ", i)
    # This is done without minimize function for easier debugging
    train_loss[i] = 0.0
    batch_count = 0
    for (x, y) in dtrn
		# This is for testing ResNet with 3 input channels
		x = cat(x, x, x, dims=3)
		loss = @diff loss_nll(model, x, y)
        train_loss[i] += value(loss)
        batch_count += 1
        for p in params(model)
            g = grad(loss, p)
            update!(value(p), g, p.opt)
        end
    end
    train_loss[i] /= batch_count
    println("Train epoch loss: ", train_loss[i])
    val_batch_count = 0
    val_acc[i] = validate(model, dtst)
    println("Accuracy :", val_acc[i])
	flush(stdout)
end
