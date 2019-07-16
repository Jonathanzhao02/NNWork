#install & load packages
install.packages(c("keras", "doParallel", "foreach"))
library(keras)
library(doParallel)
library(foreach)

#install Keras
install_keras()

#obtain data
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

#paramaters
batchSize <- 5
popSize <- 1000
generations <- 1000000 %/% (popSize * batchSize)
parentFrac <- 0.4
crossoverChance <- 0.5
mutationChance <- 0.01
mutationMagnitude <- 0.05
inheritanceDecayRate <- 0.8
hiddenUnits <- 128
networkSize <- hiddenUnits * 784 + hiddenUnits * 10 + hiddenUnits + 10
weightInitMagnitude <- 0.05
elitePortion <- 0.05
elite <- ceiling(elitePortion * popSize)

initNetwork <- function(){
  network <<- keras_model_sequential() %>%
    layer_dense(units = hiddenUnits,
                activation = "relu",
                input_shape = c(28 * 28)) %>%
    layer_dense(units = 10,
                activation = "softmax")
  
  network %>% compile(
    optimizer = optimizer_sgd(),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
}

initWeights <- function(){
  l1 <- array(dim = c(784,hiddenUnits))
  l1[1:784,1:hiddenUnits] <- rnorm(784 * hiddenUnits, sd = weightInitMagnitude)
  l2 <- array(dim = c(hiddenUnits))
  l2[1:hiddenUnits] <- rnorm(hiddenUnits, sd = weightInitMagnitude)
  l3 <- array(dim = c(hiddenUnits,10))
  l3[1:hiddenUnits,1:10] <- rnorm(hiddenUnits * 10, sd = weightInitMagnitude)
  l4 <- array(dim = c(10))
  l4[1:10] <- rnorm(10, sd = weightInitMagnitude)
  return(list(l1, l2, l3, l4))
}

getTestData <- function(numSamples){
  indices <- sample(1:70000, size = numSamples)
  
  if(numSamples < 10 && exists("prev_labels")){
  
    while(contains(prev_labels, numSamples, indices)){
      indices <- sample(1:70000, size = numSamples)
    }
  
  }
    
  test_images <<- all_images[indices,]
  test_labels <<- all_labels[indices,]
  prev_labels <<- which(1 %in% test_labels)
}

contains <- function(prev_labels, numSamples, indices){
  
  for(i in 1:numSamples){
    intY <- prev_labels[i] %/% numSamples
    yVals <- all_labels[indices,intY]
    
    if(1 %in% yVals){
      return(TRUE)
    }
    
  }
  
  return(FALSE)
}

createGen <- function(){
  exportedVars <- c(
    "getFitChild",
    "createChild",
    "crossover",
    "mutate",
    "prev_gen",
    "popSize",
    "parentFrac",
    "networkSize",
    "hiddenUnits",
    "crossoverChance",
    "mutationChance",
    "mutationMagnitude")
  
  vals <- foreach(i = 1:(popSize - elite), .combine = 'c', .export = exportedVars) %dopar% {
    parent_1 <- getFitChild()
    parent_2 <- getFitChild()
    avg_fitness <- (parent_1$fitness + parent_2$fitness) / 2
    result <- list(list(createChild(parent_1$val, parent_2$val)), avg_fitness)
  }
  
  for(i in 1:(popSize - elite)){
    current_gen[i,]$val <<- vals[[2 * i - 1]]
    current_gen[i,]$fitness <<- vals[[2 * i]]
  }
  
  for(i in 1:elite){
    current_gen[(popSize - i + 1),] <<- prev_gen[i,]
  }
  
}

getFitChild <- function(){
  fitness_sum <- 0
  running_fitness_sum <- 0
  
  for(i in 1:(popSize * parentFrac)){
    fitness_sum <- fitness_sum + prev_gen[i,]$fitness
  }
  
  chosen <- fitness_sum * runif(1)
  
  for(i in 1:(popSize * parentFrac)){
    
    if(running_fitness_sum + prev_gen[i,]$fitness >= chosen){
      return(prev_gen[i,])
    } else{
      running_fitness_sum <- running_fitness_sum + prev_gen[i,]$fitness
    }
    
  }
  
}

createChild <- function(parent_1, parent_2){
  
  while(identical(parent_1, parent_2)){
    parent_2 <- getFitChild()$val
  }
  
  #vectorizing parents
  parent_1 <- unlist(parent_1[[1]][1:4])
  parent_2 <- unlist(parent_2[[1]][1:4])
  
  #crossover and mutation
  child <- array(dim = c(networkSize))
  
  if(runif(1) < crossoverChance){
    child <- crossover(parent_1, parent_2)
  } else{
    child <- parent_1
  }
  
  child <- mutate(child)
  
  #reshapes back into separate layer weights and biases
  nextIndex <- 0
  l1 <- array(child[1:(nextIndex <- nextIndex + 784 * hiddenUnits)], dim = c(784,hiddenUnits))
  l2 <- array(child[(nextIndex + 1):(nextIndex <- nextIndex + hiddenUnits)], dim = c(hiddenUnits))
  l3 <- array(child[(nextIndex + 1):(nextIndex <- nextIndex + 10 * hiddenUnits)], dim = c(hiddenUnits,10))
  l4 <- array(child[(nextIndex + 1):(nextIndex <- nextIndex + 10)], dim = c(10))
  
  return(list(l1, l2, l3, l4))
}

crossover <- function(parent_1, parent_2){
  child <- array(dim = c(networkSize))
  indices <- sample(1:networkSize, size = networkSize %/% 2)
  child[indices] <- parent_1[indices]
  child[-indices] <- parent_2[-indices]
  return(child)
}

mutate <- function(child){
  condition <- (runif(networkSize) < mutationChance)
  indices <- which(condition == FALSE)
  nums <- rnorm(networkSize, sd = mutationMagnitude)
  nums[indices] <- 0
  child <- child + nums
  return(child)
}

#Execution code

#Setup parallel processing
stopCluster(cluster)
registerDoSEQ()
cores <- detectCores()
cluster <- makeCluster(cores[1] - 1, outfile = "clusterErrors.txt")
registerDoParallel(cluster)

#Initialize network and dataframe structure
initNetwork()
df <- data.frame(val = 0, fitness = 0)
df[1:networkSize,]$fitness <- 0
current_gen <- df
best_fitness <- list()
genCount <- 1

#Generation loop
while(genCount <= generations){
  getTestData(batchSize)
  
  #Children loop
  for(j in 1:popSize){
    
    if(genCount == 1){
      weights <- list(initWeights())
    } else{
      weights <- current_gen[j,]$val
    }
    
    set_weights(network, c(weights[[1]][1:4]))
    metrics <- evaluate(network, test_images, test_labels)
    current_gen[j,]$val <- weights
    current_gen[j,]$fitness <- inheritanceDecayRate * current_gen[j,]$fitness + ((metrics$acc * 2 + 1) / (sqrt(metrics$loss)))
    print(cat("Generation", genCount, "Child", j, "Fitness:", current_gen[j,]$fitness))
  }
  
  current_gen <- current_gen[order(current_gen$fitness, decreasing = TRUE),]
  best_fitness[[genCount]] <- current_gen[1,]
  
  if(genCount %% 10 == 0){
    save.image(file = paste("gen", genCount, ".Rdata", sep = ""))
  }
  
  prev_gen <- current_gen
  current_gen <- df
  createGen()
  genCount <- genCount + 1
}

#Plot results
history <- array(dim = c(length(best_fitness)))

for(i in 1:length(best_fitness)){
  set_weights(network, c(best_fitness[[i]]$val[[1]][1:4]))
  metrics <- evaluate(network, all_images, all_labels)
  history[i] <- metrics$acc
}

plot(1:length(best_fitness), history, ylim = c(0, 1))
save.image(file = "genFINAL.Rdata")

#testing
proctime <- system.time({
  
})[3]
proctime