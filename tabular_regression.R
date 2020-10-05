library(torch)
library(palmerpenguins)

data("penguins")

penguins <- na.omit(penguins)

## Train/Test Split 
train_idx <- sample(nrow(penguins), nrow(penguins)*.7)

features <- c('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'island')

response <- 'body_mass_g'

df_dataset <- dataset(
  name = "Penguins",
  
  # the input data to your dataset goes in the initialize function.
  # our dataset will take a dataframe and the name of the response
  # variable.
  initialize = function(df, feature_variables, response_variable) {
    
    # conveniently, the categorical data are already factors
    df$species <- as.numeric(df$species)
    df$island <- as.numeric(df$island)
    
    self$df <- df[, feature_variables]
    self$response_variable <- df[[response_variable]]
  },
  
  # the .getitem method takes an index as input and returns the
  # corresponding item from the dataset.
  # the index could be anything. the dataframe could have many
  # rows for each index and the .getitem method would do some
  # kind of aggregation before returning the element.
  # in our case the index will be a row of the data.frame,
  
  .getitem = function(index) {
    response <- torch_tensor(self$response_variable[index], dtype = torch_float())
    x <- torch_tensor(as.numeric(self$df[index,]))
    
    # note that the dataloaders will automatically stack tensors
    # creating a new dimension
    list(x = x, y = response)
  },
  
  # It's optional, but helpful to define the .length method returning 
  # the number of elements in the dataset. This is needed if you want 
  # to shuffle your dataset.
  .length = function() {
    length(self$response_variable)
  }
  
)


penguins_train <- df_dataset(penguins[train_idx,], 
                             feature_variables = features, 
                             response_variable = response)

penguins_test <- df_dataset(penguins[-train_idx,],
                            feature_variables = features, 
                            response_variable = response)

penguins_train$.getitem(100)


dl_train <- dataloader(penguins_train, batch_size = 10, shuffle = TRUE)

dl_test <-  dataloader(penguins_test, batch_size = 10)

for(batch in enumerate(dl_train)) {
  cat("X size:  ")
  print(batch[[1]]$size())
  cat("Y size:  ")
  print(batch[[2]]$size())
}

iter <- dl_train$.iter()
iter$.next()


net <- nn_module(
  "PenguinNet",
  initialize = function() {
    self$fc1 <- nn_linear(length(features), 16)
    self$fc2 <- nn_linear(16, 8)
    self$fc3 <- nn_linear(8, 1)
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      nnf_relu() %>% 
      self$fc2() %>% 
      nnf_relu() %>%
      self$fc3() 
  }
)

model <- net()

optimizer <- optim_adam(model$parameters)

for (epoch in 1:10) {
  
  l <- c()
  
  for (b in enumerate(dl_train)) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_mse_loss(output,b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
  
}

# Put the model into eval mode
model$eval()

test_losses <- c()

for (b in enumerate(dl_test)) {
  output <- model(b[[1]])
  loss <- nnf_mse_loss(output, b[[2]])
  test_losses <- c(test_losses, loss$item())

}

mean(test_losses)



# Placeholder vector for predictions
preds = c()

for (b in enumerate(dl_test)) {
  
  # get predictions
  output <- model(b[[1]])
  
  # convert to vector and append
  predicted = output$data() %>% as.array() %>% .[,1]
  preds <- c(preds, predicted)
  
  
}


head(preds)
