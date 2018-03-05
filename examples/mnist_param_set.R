param.set <- makeParamSet(
  makeDiscreteParam("batch_size", values = c(16, 32, 64, 128, 256, 512)),
  makeIntegerParam("epochs", lower = 5, upper = 500)
)
