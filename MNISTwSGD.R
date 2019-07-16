#install & load packages
install.packages("keras")
library(keras)
use_condaenv("r-tensorflow")
install_keras(tensorflow="gpu")

library(tensorflow)
install_tensorflow(version="gpu")

#paramaters
batchSize <- 400
learningRate <- 0.5
hiddenUnits <- 128

mnist <- dataset_mnist()

all_images <- array(dim = c(70000, 28, 28))
all_images[1:60000,,] <- mnist$train$x
all_images[60001:70000,,] <- mnist$test$x
all_images <- array_reshape(all_images, c(70000, 28 * 28))
all_images <- all_images / 255

all_labels <- array(dim = c(70000))
all_labels[1:60000] <- mnist$train$y
all_labels[60001:70000] <- mnist$test$y
all_labels <- to_categorical(all_labels)

train_images <- c()
train_labels <- c()
test_images <- c()
test_labels <- c()
validation_images <- c()
validation_labels <- c()

randomizeData <- function(){
  indices <- sample(1:70000, size = 50000)
  
  train_images <<- all_images[indices,]
  train_labels <<- all_labels[indices,]
  
  misc_images <- all_images[-indices,]
  misc_labels <- all_labels[-indices,]
  
  indices <- sample(1:20000, size = 10000)
  
  test_images <<- misc_images[indices,]
  test_labels <<- misc_labels[indices,]
  validation_images <<- misc_images[-indices,]
  validation_labels <<- misc_labels[-indices,]
}

initNetwork <- function(){
  network <- keras_model_sequential() %>%
    layer_dense(units = hiddenUnits,
                activation = "relu",
                input_shape = c(28 * 28),
                kernel_initializer = "random_normal",
                bias_initializer = "random_normal") %>%
    layer_dense(units = 10,
                activation = "softmax",
                kernel_initializer = "random_normal",
                bias_initializer = "random_normal")
  
  network %>% compile(
    optimizer = optimizer_sgd(learningRate),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
}

total_time <- c(5)
total_history <- list()

for(i in 1:10){
  randomizeData()
  network <- initNetwork()
  old_time <- proc.time()
  history <- network %>% fit(
    train_images,
    train_labels,
    epochs = 20,
    batch_size = batchSize,
    validation_data = list(validation_images, validation_labels))
  total_time <- total_time + proc.time() - old_time
  metrics <- network %>% evaluate(test_images, test_labels)
  total_history[[i]] <- history
  total_history[[i]]$metrics$final_val <- metrics
  print(metrics)
}

summary(network)
total_time <- total_time / i
print(total_time)

# Plots & saves loss
mod_history <- array(0, dim = c(20))

for(i in 1:10){
  mod_history <- mod_history + total_history[[i]]$metrics$loss
  write(x = total_history[[i]]$metrics$loss, file = paste("loss", i, ".txt", sep = ""))
}

mod_history <- mod_history / 10
png(filename = "SGD_loss.png", width = 720, height = 400)
plot(1:20, mod_history, xlab = "Epochs", ylab = "Loss")
dev.off()

# Plots & saves accuracy
mod_history <- array(0, dim = c(20))

for(i in 1:10){
  mod_history <- mod_history + total_history[[i]]$metrics$acc
  write(x = total_history[[i]]$metrics$acc, file = paste("acc", i, ".txt", sep = ""))
}

mod_history <- mod_history / 10
png(filename = "SGD_acc.png", width = 720, height = 400)
plot(1:20, mod_history, xlab = "Epochs", ylab = "Accuracy")
dev.off()

# Plots & saves validation loss
mod_history <- array(0, dim = c(20))

for(i in 1:10){
  mod_history <- mod_history + total_history[[i]]$metrics$val_loss
  write(x = total_history[[i]]$metrics$val_loss, file = paste("val_loss", i, ".txt", sep = ""))
}

mod_history <- mod_history / 10
png(filename = "SGD_val_loss.png", width = 720, height = 400)
plot(1:20, mod_history, xlab = "Epochs", ylab = "Validation Loss")
dev.off()

# Plots & saves validation accuracy
mod_history <- array(0, dim = c(20))

for(i in 1:10){
  mod_history <- mod_history + total_history[[i]]$metrics$val_acc
  write(x = total_history[[i]]$metrics$val_acc, file = paste("val_acc", i, ".txt", sep = ""))
}

mod_history <- mod_history / 10
png(filename = "SGD_val_acc.png", width = 720, height = 400)
plot(1:20, mod_history, xlab = "Epochs", ylab = "Validation Accuracy")
dev.off()

# Time elapsed
#user  system elapsed 
#23.902   6.521  26.789 