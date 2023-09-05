###########################  BayCANN   #########################################
#
#  Objective: Script to perform an emulator-based Bayesian calibration on MISCAN-Colon 
########################### <<<<<>>>>> ##############################################

#### 1.Libraries and functions  ==================================================

library(keras)
library(rstan)
library(reshape2)
library(tidyverse)
library(ggplot2)
library(doParallel)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


###### 1.1 Load baycann functions =================================================
#* Clean environment
rm(list = ls())
source("baycann_functions.R")

###### 1.2 Set version and parameters of BayCANN =================================================


BayCANN_version  <- "MISCAN_v2_20220716_1637"  #Version for paper and SMDM

folder <- paste0("output/BayCANN_versions/",BayCANN_version)

#Load the parameters saved for this version
path_params <- paste0("output/BayCANN_versions/",
                      BayCANN_version,
                      "/parameters_",
                      BayCANN_version, ".RData")
load(path_params)
list2env(param_BayCANN, envir = .GlobalEnv)

#### 2. General parameters ========================================================
# The parameters in this section are already loaded but we keep them for transparency purposes 

###### 2.1 parameters for Data preparation 
scale_type=1  ## 1: for scale from -1 to 1; 2: for standardization ; 3: for scale from 0 to 1
set.seed(1234)

###### 2.1 parameters for ANN 
verbose          <- 0
n_batch_size     <- 2000
n_epochs         <- 15000
patience         <- 10000
n_hidden_nodes   <- 114
n_hidden_layers  <- 4
activation_fun   <- 'tanh'
init_W=NULL

# Other options to initialize weights
#init_W=initializer_random_uniform(minval = -0.7, maxval = 0.7,seed = 2312)   ###initialization of weights with uniform distribution
#init_W=initializer_random_normal(mean = 0, stddev = 0.1, seed = 2312)  ###initialization of weights with normal distribution


###### 2.2 parameters for Bayesian calibration
n_iter    <- 300000
n_thin    <- 100
n_chains  <- 4

#### 3. Pre-processing actions  ===========================================

Normalize_inputs    <- FALSE
Normalize_outputs   <- FALSE
Scale_inputs        <- TRUE  
Scale_outputs       <- TRUE  
Remove_outliers     <- TRUE  
Standardize_targets <- FALSE
Saved_data          <- FALSE
Selected_targets    <- FALSE

#### 4. Load the LHS data for the simulations =======================
params_file      <- "data-raw/20220622_MISCANColon_LHS/SimulatedParameters_20220622_0147UTC.csv"
data_sim_param <- read.csv(params_file)
selected_params  <- c(5:(length(data_sim_param)-1))

outputs_file     <- "data-raw/20220622_MISCANColon_LHS/SimulatedTargets_20220622_0147UTC.csv"
data_sim_target <- read.csv(outputs_file)
selected_targets <- c(4:(length(data_sim_target)))

targets_file     <- "data-raw/20220622_MISCANColon_LHS/Targets_MISCAN_v2.0.csv"


###### 4.1 Inputs ----
data_sim_param <- read.csv(params_file)
data_sim_param <- data_sim_param[,selected_params]
data_sim_param <- as.matrix(data_sim_param)
###### 4.2 Outputs ----
data_sim_target <- read.csv(outputs_file)
data_sim_target <- data_sim_target[,selected_targets]
data_sim_target <- as.matrix(data_sim_target)
###### 4.3 Targets ----


#### 5. Removing outliers from model outputs ####

if (Remove_outliers) {

  vec_maxs  <- apply(data_sim_target, 1, max)
  vec_999   <- (vec_maxs==99999)
  data_sim_param  <- data_sim_param[!vec_999,]
  data_sim_target <- data_sim_target[!vec_999,]
}



#### 7. Train/test partition ======================================================

require(caTools)
set.seed(1223)

train.rows<-sample.split(data_sim_param[,1],SplitRatio=0.8)

data_sim_param_train  <- data_sim_param[train.rows,]
data_sim_param_test   <- data_sim_param[!train.rows,]

data_sim_target_train <- data_sim_target[train.rows,]
data_sim_target_test  <- data_sim_target[!train.rows,]

prepared_data <- prepare_data(xtrain = data_sim_param_train,
                              ytrain = data_sim_target_train,
                              xtest  = data_sim_param_test,
                              ytest  = data_sim_target_test,
                              scale  = scale_type)

list2env(prepared_data, envir = .GlobalEnv)


####  9. Load the targets and their se ============================================

data_true_targets  <- read.csv(targets_file)

# If we select an specific list of targets
if(Selected_targets) {
  data_true_targets  <- data_true_targets[(data_true_targets$target_names%in%Selected_MISCAN_R_Targets),]   
}

true_targets_mean  <- data_true_targets$targets
true_targets_upper <- data_true_targets$stopping_upper_bounds
true_targets_lower <- data_true_targets$stopping_lower_bounds
true_targets_se   <- (true_targets_upper - true_targets_lower)/(2*1.96)


#### 10. Scale the targets and their SE  ####

if (scale_type==1) {
  y_targets <- 2 * (true_targets_mean - ymins) / (ymaxs - ymins) - 1   ## range from -1 to 1
  y_targets_se <- 2 * (true_targets_se) / (ymaxs - ymins)
}

if (scale_type==2) {
  y_targets <- (true_targets_mean - ymeans)/ysds   ## Standardization
  y_targets_se <-(true_targets_se)*(sqrt(100))/ysds
}

if (scale_type==3) {
  y_targets <- (true_targets_mean - ymins) / (ymaxs - ymins)   ## range from 0 to 1
  y_targets_se <-(true_targets_se) / (ymaxs - ymins)
}

y_targets <- t(as.matrix(y_targets))
y_targets_se <- t(as.matrix(y_targets_se))   



#### 11. Keras Section BayCANN SimCRC ==============================================

# File name of keras model

path_keras_model <- paste0(folder,"/model_keras_",BayCANN_version,".h5")    ##File path for the compiled model


model <- keras_model_sequential()
mdl_string <- paste("model %>% layer_dense(units = n_hidden_nodes, kernel_initializer=init_W, activation = activation_fun, input_shape = n_inputs) %>%",
                    paste(rep(x = "layer_dense(units = n_hidden_nodes, activation = activation_fun) %>%",
                              (n_hidden_layers)), collapse = " "),
                    "layer_dense(units = n_outputs)")

eval(parse(text = mdl_string))

summary(model)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'  ,
  metrics = list('mae',"accuracy")
)

keras.time <- proc.time()

history <- model %>% fit(
  xtrain_scaled, ytrain_scaled,
  epochs = n_epochs,
  batch_size = n_batch_size,
  validation_data = list(xtest_scaled, ytest_scaled),
  verbose = verbose,
  callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    verbose = 0,
    restore_best_weights = TRUE
  )
)

t_training <- proc.time() - keras.time #keras ann fitting time

t_training <- t_training[3]/60

acc_err<-model%>%evaluate(xtest_scaled,ytest_scaled) # Model performance evaluation
metric_loss      <- acc_err[1]
metric_mae       <- acc_err[2]
metric_accuracy  <- acc_err[3]

save_model_hdf5(model,path_keras_model)  #Save the model
model <- load_model_hdf5(path_keras_model)


###### 11.4 History Graph ####

plot(history)   #Plot loss function and accuracy function

###### 11.5 Prediction Graph  ####

pred <- model %>% predict(xtest_scaled)
ytest_scaled_pred <- data.frame(pred)
colnames(ytest_scaled_pred) <- y_names
head(ytest_scaled_pred)    #

ann_valid <- rbind(data.frame(sim = 1:n_test, ytest_scaled, type = "model"),
                   data.frame(sim = 1:n_test, ytest_scaled_pred, type = "pred"))
ann_valid_transpose <- ann_valid %>%
  pivot_longer(cols = -c(sim, type)) %>%
  pivot_wider(id_cols = c(sim, name), names_from = type, values_from = value)

##Partition of validation data for Graph (3 parts)
n_partition <-4
n_part_bach <-floor(n_outputs/n_partition)

ann_valid_transpose <- arrange(ann_valid_transpose,desc(name))

ann_valid_transpose1 <- ann_valid_transpose[(1):(n_part_bach*n_test),]
ann_valid_transpose2 <- ann_valid_transpose[(n_part_bach*n_test+1):(2*n_part_bach*n_test),]
ann_valid_transpose3 <- ann_valid_transpose[(2*n_part_bach*n_test+1):(3*n_part_bach*n_test),]
ann_valid_transpose4 <- ann_valid_transpose[(3*n_part_bach*n_test+1):dim(ann_valid_transpose)[1],]

#part 1
ggplot(data = ann_valid_transpose1, aes(x = model, y = pred)) +
  geom_point(alpha = 0.5, color = "tomato") +
  facet_wrap(~name, scales="free",  ncol = 7) +
  xlab("Model outputs (scaled)") +
  ylab("ANN predictions (scaled)") +
  #coord_equal() +
  theme_bw()

#part 2
ggplot(data = ann_valid_transpose2, aes(x = model, y = pred)) +
  geom_point(alpha = 0.5, color = "tomato") +
  facet_wrap(~name, scales="free", ncol = 7) +
  xlab("Model outputs (scaled)") +
  ylab("ANN predictions (scaled)") +
  #coord_equal() +
  theme_bw()

#part 3
ggplot(data = ann_valid_transpose3, aes(x = model, y = pred)) +
  geom_point(alpha = 0.5, color = "tomato") +
  facet_wrap(~name, scales="free", ncol = 7) +
  xlab("Model outputs (scaled)") +
  ylab("ANN predictions (scaled)") +
  #coord_equal() +
  theme_bw()

#part 4
ggplot(data = ann_valid_transpose4, aes(x = model, y = pred)) +
  geom_point(alpha = 0.5, color = "tomato") +
  facet_wrap(~name, scales="free", ncol = 7) +
  xlab("Model outputs (scaled)") +
  ylab("ANN predictions (scaled)") +
  #coord_equal() +
  theme_bw()


#### 12. Stan section ==============================================================

path_posterior <- paste0(folder,"/calibrated_posteriors_",BayCANN_version,".csv")

weights <- get_weights(model) #get ANN weights

n_hidden_layers <- length(weights)/2-2    #Removing bias layers and input and output layers  (Carlos P)
n_hidden_nodes  <- ncol(weights[[1]])     #Get number of hidden nodes from the firs layers (Carlos P)

# pass the weights and biases to Stan for Bayesian calibration
n_layers <- length(weights)
weight_first <- weights[[1]]
beta_first <- 1 %*% weights[[2]]
weight_last <- weights[[n_layers-1]]
beta_last <- 1 %*% weights[[n_layers]]
weight_middle <- array(0, c(n_hidden_layers, n_hidden_nodes, n_hidden_nodes))
beta_middle <- array(0, c(n_hidden_layers, 1, n_hidden_nodes))
for (l in 1:n_hidden_layers){
  weight_middle[l,,] <- weights[[l*2+1]]
  beta_middle[l,,] <- weights[[l*2+2]]
}

###Información para inferencia en Stan
stan.dat=list(
  num_hidden_nodes = n_hidden_nodes,
  num_hidden_layers= n_hidden_layers,
  num_inputs=n_inputs,
  num_outputs=n_outputs,
  num_targets=1,
  y_targets = y_targets,
  y_targets_se = y_targets_se,
  beta_first = beta_first,
  beta_middle = beta_middle,
  beta_last = beta_last,
  weight_first = weight_first,
  weight_middle = weight_middle,
  weight_last = weight_last)

# Select the stan file based on data transformation

if (Normalize_inputs) {
  file_perceptron <- "post_multi_perceptron_normal.stan"
} else {
  file_perceptron <- "post_multi_perceptron.stan"  
}

# Run stan file
stan.time <- proc.time()
m <- stan(file = file_perceptron,
          data = stan.dat,
          iter = n_iter,
          chains = n_chains,
          thin = n_thin,
          pars = c("Xq"),
          warmup = floor(n_iter/2),   ## (cp)
          seed = 12345) #for reproducibility. R's set.seed() will not work for stan
t_calibration <- proc.time() - stan.time # stan sampling time
t_calibration <- t_calibration[3] / 60
summary(m)

path_stan_model <- paste0(folder,"/stan_model_", BayCANN_version,".rds")

param_names    <- colnames(data_sim_param)

names(m)[1:n_inputs] <- param_names

saveRDS(m, path_stan_model)

m <- readRDS(path_stan_model)
param_names    <- colnames(data_sim_param)
names(m)[1:n_inputs] <- param_names

###### 12.1 Stan Diagnose  ----

stan_trace(m,pars=param_names,inc_warmup = FALSE)

stan_plot(m,pars=param_names, point_est = "mean", show_density = TRUE, fill_color = "maroon", ncol=2)

stan_hist(m,pars=param_names, inc_warmup = FALSE)

stan_dens(m,pars=param_names, inc_warmup = FALSE, separate_chains=TRUE)

stan_dens(m,pars=param_names, inc_warmup = FALSE, separate_chains=FALSE)

stan_ac(m,pars=param_names[1:15], inc_warmup = FALSE, separate_chains=TRUE)
stan_ac(m,pars=param_names[16:37], inc_warmup = FALSE, separate_chains=TRUE)

stan_rhat(m,pars=param_names)          # Rhat statistic 
stan_par(m,par=param_names[1])         # Mean metrop. acceptances, sample step size
stan_ess(m,pars=param_names)           # Effective sample size / Sample size
stan_mcse(m,pars=param_names)          # Monte Carlo SE / Posterior SD
stan_diag(m,)

###### 12.2 Stan extraction  ----
params <- rstan::extract(m, permuted=TRUE, inc_warmup = FALSE)
lp <- params$lp__
Xq <- params$Xq
Xq_df = as.data.frame(Xq)


# Scale the posteriors
if (Scale_inputs) {
  Xq_unscaled <- unscale_data(Xq_df, vec.mins = xmins, vec.maxs = xmaxs, vec.means = xmeans, vec.sds = xsds, type = scale_type)
} else {
  Xq_unscaled <- Xq_df
}


Xq_lp <- cbind(Xq_unscaled, lp)
# Save the unscaled posterior samples
write.csv(Xq_lp,
          file = path_posterior,
          row.names = FALSE)

cal_mdl_1 <- path_posterior
### Load ANN posterior
Xq_lp <- read.csv(file = cal_mdl_1)
n_col <- ncol(Xq_lp)
lp <- Xq_lp[, n_col]
Xq_unscaled <- Xq_lp[, -n_col]
map_baycann <- Xq_unscaled[which.max(lp), ]     ### MAP for first BayCANN model
df_post_ann <- read.csv(file = cal_mdl_1)[, -n_col]
colnames(df_post_ann) <- x_names


##correlation graph
library(GGally)

df_post <- data.frame(Xq_unscaled)
colnames(df_post) <- x_names
df_post_long <- reshape2::melt(df_post,
                               variable.name = "Parameter")

df_post_long$Parameter <- factor(df_post_long$Parameter,
                                 levels = levels(df_post_long$Parameter),
                                 ordered = TRUE)

gg_calib_post_pair_corr <- GGally::ggpairs(df_post,
                                           upper = list(continuous = wrap("cor",
                                                                          color = "black",
                                                                          size = 5)),
                                           diag = list(continuous = wrap("barDiag",
                                                                         alpha = 0.8)),
                                           lower = list(continuous = wrap("points",
                                                                          alpha = 0.3,
                                                                          size = 0.5)),
                                           labeller = "label_parsed") +
  theme_bw(base_size = 18) +
  theme(axis.title.x = element_blank(),
        axis.text.x  = element_text(size=6),
        axis.title.y = element_blank(),
        axis.text.y  = element_blank(),
        axis.ticks.y = element_blank(),
        strip.background = element_rect(fill = "white",
                                        color = "white"),
        strip.text = element_text(hjust = 0))
gg_calib_post_pair_corr

ggsave(filename = paste0(folder,"/fig_posterior_distribution_pairwise_corr_",BayCANN_version,".png"),
       gg_calib_post_pair_corr,
       width = 36, height = 24)


#### Prior and prior graph
n_samp <- 1000
df_samp_prior <- melt(cbind(Distribution = "Prior",
                            as.data.frame(data_sim_param_train[1:n_samp, ])),
                      variable.name = "Parameter")

df_samp_post_ann   <- melt(cbind(Distribution = "Posterior BayCANN",
                                 as.data.frame(df_post_ann[1:n_samp, ])),
                           variable.name = "Parameter")


df_samp_prior_post <- rbind(df_samp_prior,
                            df_samp_post_ann)
df_samp_prior_post$Distribution <- ordered(df_samp_prior_post$Distribution,
                                           levels = c("Prior",
                                                      "Posterior BayCANN"))



df_samp_prior_post$Parameter <- factor(df_samp_prior_post$Parameter,
                                       levels = levels(df_samp_prior_post$Parameter),
                                       ordered = TRUE)

df_maps_n_true_params <- data.frame(Type = ordered(rep(c( "BayCANN MAP"), each = n_inputs),
                                                   levels = c("BayCANN MAP")),
                                    value = c(t(map_baycann)))
df_maps_n_true_params


### Plot priors and ANN and IMIS posteriors


df_maps_n_true_params$Parameter<-as.factor(x_names)


library(dampack)

gg_ann_vs_imis <- ggplot(df_samp_prior_post,
                         aes(x = value, y = ..density.., fill = Distribution)) +
  facet_wrap(~Parameter, scales = "free",
             ncol = 4,
             labeller = label_parsed) +
  scale_x_continuous(breaks = number_ticks(5)) +
  scale_color_manual("", values = c("black", "navy blue", "tomato","green")) +
  geom_density(alpha=0.5) +
  theme_bw(base_size = 16) +
  guides(fill = guide_legend(title = "", order = 1),
         linetype = guide_legend(title = "", order = 2),
         color = guide_legend(title = "", order = 2)) +
  theme(legend.position = "bottom",
        legend.box = "vertical",
        legend.margin=margin(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        strip.background = element_rect(fill = "white",
                                        color = "white"),
        strip.text = element_text(hjust = 0))
gg_ann_vs_imis
ggsave(gg_ann_vs_imis,
       filename = paste0(folder,"/fig_posterior_calibrated_distribution.png"),
       width = 12, height = 12)


# 13. Internal validation  -------------------------------------------------

###### 13.1 Load Stan file -----
m <- readRDS(path_stan_model) 

#Extraction of parameters 
params <- rstan::extract(m)
Xq <- params$Xq
Xq_df = as.data.frame(Xq)
colnames(Xq_df) <- x_names
Xq_df <- as.matrix(Xq_df)

pred_posterior <- model %>% predict(Xq_df)
pred_posterior <- data.frame(pred_posterior)
colnames(pred_posterior) <- y_names

if (Scale_outputs) {
  pred_posterior_unsc <- unscale_data(pred_posterior, vec.mins = ymins, vec.maxs = ymaxs, vec.means = ymeans, vec.sds = ysds, type = scale_type)
}

path_validation_ANN <- paste0(folder,"/prediction_ANN_posteriors_",BayCANN_version,".csv")
write.csv(pred_posterior_unsc,
          file = path_validation_ANN,
          row.names = FALSE)

# 14. Save BayCANN parameters  -------------------------------------------------

##Save BayCANN parameters
path_baycann_params <- paste0(folder,"/parameters_",BayCANN_version,".RData")
param_BayCANN <- list(BayCANN_version    = BayCANN_version, 
                      scale_type         = scale_type,
                      scale_type         = scale_type, 
                      verbose            = verbose,
                      n_batch_size       = n_batch_size,
                      n_chains           = n_chains,
                      n_epochs           = n_epochs,
                      patience           = patience,
                      n_hidden_nodes     = n_hidden_nodes,
                      n_hidden_layers    = n_hidden_layers,
                      activation_fun     = activation_fun,
                      init_W             = init_W,
                      n_iter             = n_iter,
                      n_thin             = n_thin,
                      Normalize_inputs   = Normalize_inputs,
                      Normalize_outputs  = Normalize_outputs,
                      Scale_inputs       = Scale_inputs,
                      Scale_outputs      = Scale_outputs,
                      Remove_outliers    = Remove_outliers, 
                      Standardize_targets= Standardize_targets,
                      Saved_data         = Saved_data,
                      Selected_targets   = Selected_targets,
                      params_file        = params_file,
                      selected_params    = selected_params,
                      outputs_file       = outputs_file,
                      selected_targets   = selected_targets,
                      targets_file       = targets_file,
                      path_keras_model   = path_keras_model,
                      t_training         = t_training,
                      metric_loss        = metric_loss,
                      metric_mae         = metric_mae,
                      metric_accuracy    = metric_accuracy, 
                      path_posterior     = path_posterior, 
                      file_perceptron    = file_perceptron, 
                      t_calibration      = t_calibration,
                      path_stan_model    = path_stan_model,
                      path_validation_ANN= path_validation_ANN,
                      path_baycann_params=path_baycann_params
)

save(param_BayCANN, file = path_baycann_params)
