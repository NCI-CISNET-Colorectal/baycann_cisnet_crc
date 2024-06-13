#################  BayCANN posterior distributions validation   ################
#
#  Objective: Script to validate the posterior distributions from BayCANN using  
#             the target's empirical accepted windows
########################### <<<<<>>>>> #########################################

####==================== 1. Load_packages ==================================####

library(ggplot2)
library(dplyr)
library(dampack)
library(ggpubr)
library(patchwork)
library(GGally)
library(doBy)
library(tidyverse)
library(readr)
library(readxl)
####==================== 2. Load BayCANN functions =========================####
#* Clean environment
rm(list = ls())

#* Load BayCANN function
source("baycann_functions.R")

####==================== 3. Load data =====================================####

######  3.1 SimCRC  ----


#version 7 BayCANN (SimCRC) (26/Sep/2022) (VERSION FOR PAPER)
#Outputs
data_outputs_simcrc <- read.csv("validations/SimCRC/20220926/20220926_AllLLOutput.csv")
data_outputs_simcrc <- subset(data_outputs_simcrc, 
                              select = -c(OutTag,Sex, QuitEarly, Seed))
data_outputs_simcrc <- data_outputs_simcrc[,1:110]
data_outputs_simcrc$index <- "Model"
#Targets
true_target_simcrc <- read.csv("data-raw/20220909_SimCRC_LHS/20220909_simcrc_targets.csv")
BayCANN_version <- "SimCRC_v6_20220916_2140"

# write.csv(data_outputs_simcrc,   #Save the chain 1 outputs
#           file = paste0("output/SimCRC/prediction_ANN_posteriors_SimCRC_v5_18jul22_1743_ch1.csv"),
#           row.names = FALSE)


######  3.2 MISCAN  ----

#Outputs

###### Version MISCAN 2 (11 JUL 22)  (VERSION FOR PAPER)
data_outputs_miscan <- read.csv("validations/MISCAN/targets-input_baycann_11jul22_1100.csv")
data_outputs_miscan <- data_outputs_miscan[,2:length(data_outputs_miscan)]
data_outputs_miscan$index <- "Model"
true_target_miscan <- read.csv("data-raw/20220622_MISCANColon_LHS/Targets_MISCAN_v2.0.csv")
BayCANN_version    <- "MISCAN_v2_20220716_1637"

######  3.3 CRCSPIN  ----

###### Version 1 CRC-SPIN 
data_outputs_crcspin <- read.csv("validations/CRCSPIN/20220818/SimulatedTargets_20220818_1532CDT.csv")
data_outputs_crcspin <- data_outputs_crcspin[,4:length(data_outputs_crcspin)] 
data_outputs_crcspin$index <- "Model"
true_target_crcspin <- read_excel("data-raw/20220325_CRC_SPIN/targets.xlsx")
BayCANN_version <- "CRCSPIN_v1_20221011_1345"


#* In progress

#### ====================  4. Output bounds ===============================

######  4.1 MISCAN  ----

attach(data_outputs_miscan)
collapse_median    <- summaryBy( . ~ index , FUN=c(median), data=data_outputs_miscan,keep.names=TRUE)
collapse_UB_95   <- summaryBy( . ~ index , FUN=quantile, probs = 0.975, data=data_outputs_miscan, keep.names=TRUE)
collapse_LB_95   <- summaryBy( . ~ index , FUN=quantile, probs = 0.025, data=data_outputs_miscan, keep.names=TRUE)
collapse_UB_50   <- summaryBy( . ~ index , FUN=quantile, probs = 0.75, data=data_outputs_miscan, keep.names=TRUE)
collapse_LB_50   <- summaryBy( . ~ index , FUN=quantile, probs = 0.25, data=data_outputs_miscan, keep.names=TRUE)
#collapse_median  <- summaryBy( . ~ index , FUN=c(median), data=data_outputs_miscan,keep.names=TRUE)

collapse_median  <- collapse_median %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_median")
collapse_UB_95 <- collapse_UB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_95")
collapse_LB_95 <- collapse_LB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_95")
collapse_UB_50 <- collapse_UB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_50")
collapse_LB_50 <- collapse_LB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_50")

model_outputs <- inner_join(collapse_median,collapse_UB_95, by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_95,by =c("index", "target_names")) %>% 
  inner_join(collapse_UB_50,by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_50,by =c("index", "target_names")) 

model_outputs_targets_miscan <- inner_join(model_outputs , true_target_miscan ,by=c("target_names"))  #Adding group

model_outputs_targets_miscan$n_target <- 1:dim(model_outputs_targets_miscan)[1]     #Numeric index for targets

model_outputs_targets_miscan$fitting = ifelse(model_outputs_targets_miscan$model_median <= model_outputs_targets_miscan$stopping_upper_bounds & model_outputs_targets_miscan$model_median >= model_outputs_targets_miscan$stopping_lower_bounds,
                                              yes = 1,
                                              no = 0)

table(model_outputs_targets_miscan$fitting)

detach(data_outputs_miscan)



######  4.2 SimCRC  ----

attach(data_outputs_simcrc)
collapse_mean  <- summaryBy( . ~ index , FUN=c(mean), data=data_outputs_simcrc,keep.names=TRUE)
collapse_UB_95 <- summaryBy( . ~ index , FUN=quantile, probs = 0.975, data=data_outputs_simcrc, keep.names=TRUE)
collapse_LB_95 <- summaryBy( . ~ index , FUN=quantile, probs = 0.025, data=data_outputs_simcrc, keep.names=TRUE)
collapse_UB_50 <- summaryBy( . ~ index , FUN=quantile, probs = 0.75, data=data_outputs_simcrc, keep.names=TRUE)
collapse_LB_50 <- summaryBy( . ~ index , FUN=quantile, probs = 0.25, data=data_outputs_simcrc, keep.names=TRUE)


collapse_mean  <- collapse_mean %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_mean")
collapse_UB_95 <- collapse_UB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_95")
collapse_LB_95 <- collapse_LB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_95")
collapse_UB_50 <- collapse_UB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_50")
collapse_LB_50 <- collapse_LB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_50")

model_outputs <- inner_join(collapse_mean,collapse_UB_95, by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_95,by =c("index", "target_names")) %>% 
  inner_join(collapse_UB_50,by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_50,by =c("index", "target_names")) 

model_outputs_targets_simcrc <- inner_join(model_outputs , true_target_simcrc ,by=c("target_names"))  #Adding group

model_outputs_targets_simcrc$n_target <- 1:dim(model_outputs_targets_simcrc)[1]     #Numeric index for targets

model_outputs_targets_simcrc$fitting = ifelse(model_outputs_targets_simcrc$model_mean <= model_outputs_targets_simcrc$stopping_upper_bounds & model_outputs_targets_simcrc$model_mean >= model_outputs_targets_simcrc$stopping_lower_bounds,
                                              yes = 1,
                                              no = 0)

table(model_outputs_targets_simcrc$fitting)

detach(data_outputs_simcrc)

######  4.3 CRC SPIN  ----

attach(data_outputs_crcspin)
collapse_mean  <- summaryBy( . ~ index , FUN=c(mean), data=data_outputs_crcspin,keep.names=TRUE)
collapse_UB_95 <- summaryBy( . ~ index , FUN=quantile, probs = 0.975, data=data_outputs_crcspin, keep.names=TRUE)
collapse_LB_95 <- summaryBy( . ~ index , FUN=quantile, probs = 0.025, data=data_outputs_crcspin, keep.names=TRUE)
collapse_UB_50 <- summaryBy( . ~ index , FUN=quantile, probs = 0.75, data=data_outputs_crcspin, keep.names=TRUE)
collapse_LB_50 <- summaryBy( . ~ index , FUN=quantile, probs = 0.25, data=data_outputs_crcspin, keep.names=TRUE)


collapse_mean  <- collapse_mean %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_mean")
collapse_UB_95 <- collapse_UB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_95")
collapse_LB_95 <- collapse_LB_95 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_95")
collapse_UB_50 <- collapse_UB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_UB_50")
collapse_LB_50 <- collapse_LB_50 %>%
  pivot_longer(!index, names_to = "target_names", values_to = "model_LB_50")

model_outputs <- inner_join(collapse_mean,collapse_UB_95, by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_95,by =c("index", "target_names")) %>% 
  inner_join(collapse_UB_50,by =c("index", "target_names")) %>% 
  inner_join(collapse_LB_50,by =c("index", "target_names")) 

model_outputs_targets_crcspin<- inner_join(model_outputs , true_target_crcspin ,by=c("target_names"))  #Adding group

model_outputs_targets_crcspin$n_target <- 1:dim(model_outputs_targets_crcspin)[1]     #Numeric index for targets

model_outputs_targets_crcspin$fitting = ifelse(model_outputs_targets_crcspin$model_mean <= model_outputs_targets_crcspin$stopping_upper_bounds & model_outputs_targets_crcspin$model_mean >= model_outputs_targets_crcspin$stopping_lower_bounds,
                                              yes = 1,
                                              no = 0)

table(model_outputs_targets_crcspin$fitting)

detach(data_outputs_crcspin)



#### ====================  5. Validation Graph ==============================####

######  5.1 MISCAN  ----

model_outputs_targets_miscan <- model_outputs_targets_miscan[model_outputs_targets_miscan$target_groups!="T3",]  #Removing target
model_outputs_targets_miscan$n_target <- 1:dim(model_outputs_targets_miscan)[1]


## previous work

adeno_prev_group <- c("adeno_prev_32", "adeno_prev_42","adeno_prev_52","adeno_prev_62","adeno_prev_72","adeno_prev_82","adeno_prev_92")
model_outputs_targets_miscan$target_groups <-  ifelse(model_outputs_targets_miscan$target_names %in% adeno_prev_group,"T8 adeno prev",model_outputs_targets_miscan$target_groups)

adeno_mult1_group <- c("adeno_mult_1_32","adeno_mult_1_62","adeno_mult_1_92")
model_outputs_targets_miscan$target_groups <-  ifelse(model_outputs_targets_miscan$target_names %in% adeno_mult1_group,"T8 adeno mult 1",model_outputs_targets_miscan$target_groups)

adeno_mult2_group <- c("adeno_mult_2_32","adeno_mult_2_62","adeno_mult_2_92")
model_outputs_targets_miscan$target_groups <-  ifelse(model_outputs_targets_miscan$target_names %in% adeno_mult2_group,"T8 adeno mult 2",model_outputs_targets_miscan$target_groups)

adeno_mult3_group <- c("adeno_mult_3_32","adeno_mult_3_62","adeno_mult_3_92")
model_outputs_targets_miscan$target_groups <-  ifelse(model_outputs_targets_miscan$target_names %in% adeno_mult3_group,"T8 adeno mult 3",model_outputs_targets_miscan$target_groups)



model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T1","A) Group T1: Adenoma")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T2","B) Group T2: Incidence")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T4B","C) Group T4B: Stage")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T4A","D) Group T4A: Stage")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T4C","E) Group T4C: Stage")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T5","F) Group T5: Followup incidence")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T6","G) Group T6: Screen-detected")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T7","H) Group T7: Incidence")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T8 adeno prev","I) Group T8: Prevalence")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T8 adeno mult 1","J) Group T8: Prevalence 1")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T8 adeno mult 2","K) Group T8: Prevalence 2")
model_outputs_targets_miscan$target_groups <- str_replace(model_outputs_targets_miscan$target_groups,"T8 adeno mult 3","L) Group T8: Prevalence 3")


# #model_outputs_targets_miscan$target_groups <- ordered(model_outputs_targets_miscan$target_groups,
#                                            levels = c("Prior",
#                                                       "Posterior IMABC_CRCSPIN",
#                                                       "Posterior BayCANN_CRCSPIN"))


#model_outputs_targets_miscan$target_names <- str_replace(model_outputs_targets_miscan$target_names,"adeno_","")

model_outputs_targets_miscan$target_groups_num <- unclass(as.factor(model_outputs_targets_miscan$target_groups))
model_outputs_targets_miscan$n_target2 <- model_outputs_targets_miscan$n_target+model_outputs_targets_miscan$target_groups_num  
v_target_order <- model_outputs_targets_miscan$n_target2  #vector for labels


###



#identification of categorical variables

cat_groups <- c("T3", "A) Group T1: Adenoma", "C) Group T4B: Stage", "D) Group T4A: Stage","E) Group T4C: Stage", "G) Group T6: Screen-detected","H) Group T7: Incidence")
model_outputs_targets_miscan$categorical <- ifelse(model_outputs_targets_miscan$target_groups %in% cat_groups,1,0)

#  X label 
string_label <- 'c("2"="adeno_small", '
lim=dim(model_outputs_targets_miscan)[1]
c=2
#for (i in 2:(lim-1)) {
for (i in v_target_order[2:(length(v_target_order)-1)]) {
  string_label <- paste(string_label,'"',i,'"=','"', model_outputs_targets_miscan$target_names[c],'",') 
  c <- c+1
}
string_label <- paste(string_label,'"',v_target_order[length(v_target_order)],'"=','"', model_outputs_targets_miscan$target_names[lim],'")')
x_label=eval(parse(text = string_label))

x_label <- str_replace(x_label,"adeno_","")
x_label <- str_replace(x_label,"inc_us_","")
x_label <- str_replace(x_label,"stage","")
x_label <- str_replace(x_label,"followup_inc_","")
x_label <- str_replace(x_label,"screen_det_","")
x_label <- str_replace(x_label,"int_canc_","")
x_label <- str_replace(x_label,"prev_","")
x_label <- str_replace(x_label,"mult_1_","")
x_label <- str_replace(x_label,"mult_2_","")
x_label <- str_replace(x_label,"mult_3_","")

x_label <- str_replace(x_label,"left","L")
x_label <- str_replace(x_label,"righ","R")
x_label <- str_replace(x_label,"rect","Re")

x_label <- str_replace(x_label,"0_19","10")
x_label <- str_replace(x_label,"20_29","25")
x_label <- str_replace(x_label,"30_39","35")
x_label <- str_replace(x_label,"40_49","45")
x_label <- str_replace(x_label,"50_59","55")
x_label <- str_replace(x_label,"60_69","65")
x_label <- str_replace(x_label,"70_79","75")
x_label <- str_replace(x_label,"80_99","90")

x_label <- str_replace(x_label,"0_34","17")
x_label <- str_replace(x_label,"35_44","40")
x_label <- str_replace(x_label,"45_54","50")
x_label <- str_replace(x_label,"55_64","60")
x_label <- str_replace(x_label,"65_74","70")
x_label <- str_replace(x_label,"75_84","80")
x_label <- str_replace(x_label,"85_99","92")

x_label <- str_replace(x_label,"_","-")

plot1 <- ggplot(data = model_outputs_targets_miscan, 
                aes(x    = n_target2, 
                    y    = targets, 
                    ymin = stopping_lower_bounds, 
                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="none") +
  # geom_line(data = model_outputs_targets,
  #           aes(x    = n_target,
  #               y    = targets,
  #               ymin = model_LB_95,
  #               ymax = model_UB_95), 
  #           color = "black") +
  geom_errorbar(data = model_outputs_targets_miscan[model_outputs_targets_miscan$categorical==1,],
              aes(x    = n_target2-0.08,
                  #y    = model_mean,
                  ymin = model_LB_95,
                  ymax = model_UB_95),width=.4, size=0.7, color="black", alpha = 0.7, position = "dodge2") +
  geom_ribbon(data = model_outputs_targets_miscan[model_outputs_targets_miscan$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_miscan[model_outputs_targets_miscan$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 3) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="none") +
  scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5))+
 # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 90),
        axis.title = element_text(size = 15),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - MISCAN", 
       x     = "", y = "")
  
plot1

load("figs/AbstractSMDM/plot_for_legend.RData")

plot1 <- ggarrange(plot1, 
                   ncol = 1, nrow = 1, legend = "bottom", legend.grob = get_legend(plot_for_legend))

ggsave(plot1,
       filename = paste0("output/BayCANN_versions/",BayCANN_version,"/Model_validation.jpg"),
       width = 23, height = 24)






###### 5.1.1 Figure for Abstract SDMD  ####


# Plot of MISCAN Incidence
model_outputs_targets_miscan_abstract <- model_outputs_targets_miscan[ model_outputs_targets_miscan$target_groups%in%c("B) Group T2: Incidence"),]
model_outputs_targets_miscan_abstract$target_groups <- "MISCAN: CRC incidence \nper 100,000 population"
model_outputs_targets_miscan_abstract <- model_outputs_targets_miscan_abstract[3:8,] 
ylabel <- "Rate per 100,000 people"

plot_miscan_incidence <- ggplot(data = model_outputs_targets_miscan_abstract, 
                                aes(x    = n_target2, 
                                    y    = targets, 
                                    ymin = stopping_lower_bounds, 
                                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="bottom") +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 1) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="bottom") +
  #scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(1,500))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - MISCAN", 
       x     = "Age", y = "")
#scale_color_manual( values = colors_plt )
plot_miscan_incidence

save(plot_miscan_incidence,file = "figs/AbstractSMDM/Plot_incidence_MISCAN.RData")


# Plot of MISCAN Prevalence
model_outputs_targets_miscan_abstract <- model_outputs_targets_miscan[ model_outputs_targets_miscan$target_groups%in%c("I) Group T8: Prevalence"),]
model_outputs_targets_miscan_abstract$target_groups <- "MISCAN: Prevalence of adenomas \namong population (%)"
ylabel <- "Prevalence"

model_outputs_targets_miscan_abstract$targets <- model_outputs_targets_miscan_abstract$targets*100
model_outputs_targets_miscan_abstract$stopping_lower_bounds <- model_outputs_targets_miscan_abstract$stopping_lower_bounds*100
model_outputs_targets_miscan_abstract$stopping_upper_bounds <- model_outputs_targets_miscan_abstract$stopping_upper_bounds*100
model_outputs_targets_miscan_abstract$model_LB_95 <- model_outputs_targets_miscan_abstract$model_LB_95*100
model_outputs_targets_miscan_abstract$model_UB_95 <- model_outputs_targets_miscan_abstract$model_UB_95*100
model_outputs_targets_miscan_abstract$model_LB_50 <- model_outputs_targets_miscan_abstract$model_LB_50*100
model_outputs_targets_miscan_abstract$model_UB_50 <- model_outputs_targets_miscan_abstract$model_UB_50*100


cols <- c("Target"="red","50% iqr"="black","95% iqr"="gray")

plot_miscan_prevalence <- ggplot(data = model_outputs_targets_miscan_abstract, 
                aes(x    = n_target2, 
                    y    = targets, 
                    ymin = stopping_lower_bounds, 
                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="bottom") +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 1) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="bottom") +
  #scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(0,100))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - MISCAN", 
       x     = "Age", y = "")
  #scale_color_manual( values = colors_plt )

plot_miscan_prevalence
save(plot_miscan_prevalence,file = "figs/AbstractSMDM/Plot_prevalence_MISCAN.RData")


### Plot for legends 

cols <- c("Calibration target"="red")
cols2 <- c("50% model predicted interquantile range"="gray30","95% model predicted interquantile range"="gray60")


plot_for_legend <- ggplot(data = model_outputs_targets_miscan_abstract, 
                          aes(x    = n_target2, 
                              y    = targets, 
                              ymin = stopping_lower_bounds, 
                              ymax = stopping_upper_bounds,
                          ))+ 
  geom_errorbar(aes(colour="Calibration target"), width=.4, size=0.9) +
  theme(legend.position="bottom") +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95,
                  fill = "95% model predicted interquantile range"),
              alpha = .7) +
  geom_ribbon(data = model_outputs_targets_miscan_abstract[model_outputs_targets_miscan_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50,
                  fill = "50% model predicted interquantile range"),
              alpha = .7) +
  facet_wrap(~ target_groups,scales="free", ncol = 1) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="bottom") +
  #scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(0,1))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0),
        legend.position = "bottom") +
  labs(title = "BayCANN - MISCAN", 
       x     = "Age", y = ylabel)+
  scale_color_manual( name="", values = cols ) +
  scale_fill_manual(name="",values=cols2)

plot_for_legend
save(plot_for_legend,file = "figs/AbstractSMDM/plot_for_legend.RData")


###### ======================= 5.2 SimCRC  ================================####

## previous work

prev123_1_group <- c("Prev1AdOrPreClin_age27","Prev1AdOrPreClin_age32","Prev1AdOrPreClin_age37","Prev1AdOrPreClin_age42","Prev1AdOrPreClin_age47",
                     "Prev1AdOrPreClin_age52","Prev1AdOrPreClin_age57","Prev1AdOrPreClin_age62","Prev1AdOrPreClin_age67","Prev1AdOrPreClin_age72",
                     "Prev1AdOrPreClin_age77","Prev1AdOrPreClin_age82","Prev1AdOrPreClin_age87","Prev1AdOrPreClin_age92","Prev1AdOrPreClin_age97")
model_outputs_targets_simcrc$target_groups <-  ifelse(model_outputs_targets_simcrc$target_names %in% prev123_1_group ,"Prev 123Ad 1",model_outputs_targets_simcrc$target_groups)


prev123_2_group <- c("Prev2AdOrPreClin_age27","Prev2AdOrPreClin_age32","Prev2AdOrPreClin_age37","Prev2AdOrPreClin_age42","Prev2AdOrPreClin_age47",
                     "Prev2AdOrPreClin_age52","Prev2AdOrPreClin_age57","Prev2AdOrPreClin_age62","Prev2AdOrPreClin_age67","Prev2AdOrPreClin_age72",
                     "Prev2AdOrPreClin_age77","Prev2AdOrPreClin_age82","Prev2AdOrPreClin_age87","Prev2AdOrPreClin_age92","Prev2AdOrPreClin_age97")
model_outputs_targets_simcrc$target_groups <-  ifelse(model_outputs_targets_simcrc$target_names %in% prev123_2_group ,"Prev 123Ad 2",model_outputs_targets_simcrc$target_groups)


prev123_3_group <- c("Prev3AdOrPreClin_age27","Prev3AdOrPreClin_age32","Prev3AdOrPreClin_age37","Prev3AdOrPreClin_age42","Prev3AdOrPreClin_age47",
                     "Prev3AdOrPreClin_age52","Prev3AdOrPreClin_age57","Prev3AdOrPreClin_age62","Prev3AdOrPreClin_age67","Prev3AdOrPreClin_age72",
                     "Prev3AdOrPreClin_age77","Prev3AdOrPreClin_age82","Prev3AdOrPreClin_age87","Prev3AdOrPreClin_age92","Prev3AdOrPreClin_age97")
model_outputs_targets_simcrc$target_groups <-  ifelse(model_outputs_targets_simcrc$target_names %in% prev123_3_group ,"Prev 123Ad 3",model_outputs_targets_simcrc$target_groups)


model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"Prev0Ad","A) Prevalence 0 Ad by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"Prev 123Ad 1","B) Prevalence 1 Ad by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"Prev 123Ad 2","C) Prevalence 2 Ad by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"Prev 123Ad 3","D) Prevalence 3 Ad by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"Size","E) Size")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"PrevPreclin","F) Prevalence Preclinical by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"CRCInc_P","G) Incidence P by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"CRCInc_D","H) Incidence D by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"CRCInc_R","I) Incidence R by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"CRCInc_Overall","J) Incidence Overall by age")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"StageDist_P","K) Stage P")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"StageDist_D","L) Stage D")
model_outputs_targets_simcrc$target_groups <- str_replace(model_outputs_targets_simcrc$target_groups,"StageDist_R","M) Stage R")

#identification of categorical variables

cat_groups <- c("E) Size", "K) Stage P","L) Stage D","M) Stage R")
model_outputs_targets_simcrc$categorical <- ifelse(model_outputs_targets_simcrc$target_groups %in% cat_groups,1,0)




#  X label 
string_label <- 'c("1"="Prev0AdOrPreClin_age27", '
lim=dim(model_outputs_targets_simcrc)[1]
for (i in 2:(lim-1)) {
  string_label <- paste(string_label,'"',i,'"=','"', model_outputs_targets_simcrc$target_names[i],'",')  
}
string_label <- paste(string_label,'"',lim,'"=','"', model_outputs_targets_simcrc$target_names[lim],'")')
x_label=eval(parse(text = string_label))
#string_label2 <- "c(\"1\"=\"PreClin_age27\",  \" 2 \"= \" PreClin_age32 \", \" 3 \"= \" PreClin_age37 \", \" 4 \"= \" PreClin_age42 \", \" 5 \"= \" PreClin_age47 \", \" 6 \"= \" PreClin_age52 \", \" 7 \"= \" PreClin_age57 \", \" 8 \"= \" PreClin_age62 \", \" 9 \"= \" PreClin_age67 \", \" 10 \"= \" PreClin_age72 \", \" 11 \"= \" PreClin_age77 \", \" 12 \"= \" PreClin_age82 \", \" 13 \"= \" PreClin_age87 \", \" 14 \"= \" PreClin_age92 \", \" 15 \"= \" PreClin_age97 \", \" 16 \"= \" PreClin_age27 \", \" 17 \"= \" PreClin_age32 \", \" 18 \"= \" PreClin_age37 \", \" 19 \"= \" PreClin_age42 \", \" 20 \"= \" PreClin_age47 \", \" 21 \"= \" PreClin_age52 \", \" 22 \"= \" PreClin_age57 \", \" 23 \"= \" PreClin_age62 \", \" 24 \"= \" PreClin_age67 \", \" 25 \"= \" PreClin_age72 \", \" 26 \"= \" PreClin_age77 \", \" 27 \"= \" PreClin_age82 \", \" 28 \"= \" PreClin_age87 \", \" 29 \"= \" PreClin_age92 \", \" 30 \"= \" PreClin_age97 \", \" 31 \"= \" PreClin_age27 \", \" 32 \"= \" PreClin_age32 \", \" 33 \"= \" PreClin_age37 \", \" 34 \"= \" PreClin_age42 \", \" 35 \"= \" PreClin_age47 \", \" 36 \"= \" PreClin_age52 \", \" 37 \"= \" PreClin_age57 \", \" 38 \"= \" PreClin_age62 \", \" 39 \"= \" PreClin_age67 \", \" 40 \"= \" PreClin_age72 \", \" 41 \"= \" PreClin_age77 \", \" 42 \"= \" PreClin_age82 \", \" 43 \"= \" PreClin_age87 \", \" 44 \"= \" PreClin_age92 \", \" 45 \"= \" PreClin_age97 \", \" 46 \"= \" PreClin_age27 \", \" 47 \"= \" PreClin_age32 \", \" 48 \"= \" PreClin_age37 \", \" 49 \"= \" PreClin_age42 \", \" 50 \"= \" PreClin_age47 \", \" 51 \"= \" PreClin_age52 \", \" 52 \"= \" PreClin_age57 \", \" 53 \"= \" PreClin_age62 \", \" 54 \"= \" PreClin_age67 \", \" 55 \"= \" PreClin_age72 \", \" 56 \"= \" PreClin_age77 \", \" 57 \"= \" PreClin_age82 \", \" 58 \"= \" PreClin_age87 \", \" 59 \"= \" PreClin_age92 \", \" 60 \"= \" PreClin_age97 \", \" 61 \"= \" LRGivenAdInP \", \" 62 \"= \" MRGivenAdInP \", \" 63 \"= \" HRGivenAdInP \", \" 64 \"= \" LRGivenAdInD \", \" 65 \"= \" MRGivenAdInD \", \" 66 \"= \" HRGivenAdInD \", \" 67 \"= \" LRGivenAdInR \", \" 68 \"= \" MRGivenAdInR \", \" 69 \"= \" HRGivenAdInR \", \" 70 \"= \" ages40_49 \", \" 71 \"= \" ages50_59 \", \" 72 \"= \" ages60_69 \", \" 73 \"= \" ages70_79 \", \" 74 \"= \" ages80_89 \", \" 75 \"= \" P_ages20_39 \", \" 76 \"= \" P_ages40_49 \", \" 77 \"= \" P_ages50_59 \", \" 78 \"= \" P_ages60_69 \", \" 79 \"= \" P_ages70_79 \", \" 80 \"= \" P_ages80_99 \", \" 81 \"= \" D_ages20_39 \", \" 82 \"= \" D_ages40_49 \", \" 83 \"= \" D_ages50_59 \", \" 84 \"= \" D_ages60_69 \", \" 85 \"= \" D_ages70_79 \", \" 86 \"= \" D_ages80_99 \", \" 87 \"= \" R_ages20_39 \", \" 88 \"= \" R_ages40_49 \", \" 89 \"= \" R_ages50_59 \", \" 90 \"= \" R_ages60_69 \", \" 91 \"= \" R_ages70_79 \", \" 92 \"= \" R_ages80_99 \", \" 93 \"= \" ages20_39 \", \" 94 \"= \" ages40_49 \", \" 95 \"= \" ages50_59 \", \" 96 \"= \" ages60_69 \", \" 97 \"= \" ages70_79 \", \" 98 \"= \" ages80_99 \", \" 99 \"= \" P_S1 \", \" 100 \"= \" P_S2 \", \" 101 \"= \" P_S3 \", \" 102 \"= \" P_S4 \", \" 103 \"= \" D_S1 \", \" 104 \"= \" D_S2 \", \" 105 \"= \" D_S3 \", \" 106 \"= \" D_S4 \", \" 107 \"= \" R_S1 \", \" 108 \"= \" R_S2 \", \" 109 \"= \" R_S3 \", \" 110 \"= \" R_S4 \")"
#x_label2=eval(parse(text = string_label2))  #Short labels
#model_outputs_targets <- model_outputs_targets[model_outputs_targets$target_groups!="T3",]  #Removing target

x_label <- str_replace(x_label,"CRCincPer100K_ages","")
x_label <- str_replace(x_label,"CRCincPer100K_D_ages","")
x_label <- str_replace(x_label,"CRCincPer100K_P_ages","")
x_label <- str_replace(x_label,"CRCincPer100K_R_ages","")
x_label <- str_replace(x_label,"Prev0AdOrPreClin_age","")
x_label <- str_replace(x_label,"Prev1AdOrPreClin_age","")
x_label <- str_replace(x_label,"Prev2AdOrPreClin_age","")
x_label <- str_replace(x_label,"Prev3AdOrPreClin_age","")
x_label <- str_replace(x_label,"PrevPreclinical_ages","")
x_label <- str_replace(x_label,"SizeHRGivenAdInD_wtd","HR in D")
x_label <- str_replace(x_label,"SizeHRGivenAdInP_wtd","HR in P")
x_label <- str_replace(x_label,"SizeHRGivenAdInR_wtd","HR in R")
x_label <- str_replace(x_label,"SizeLRGivenAdInD_wtd","LR in D")
x_label <- str_replace(x_label,"SizeLRGivenAdInP_wtd","LR in P")
x_label <- str_replace(x_label,"SizeLRGivenAdInR_wtd","LR in R")
x_label <- str_replace(x_label,"SizeMRGivenAdInD_wtd","MR in D")
x_label <- str_replace(x_label,"SizeMRGivenAdInP_wtd","MR in P")
x_label <- str_replace(x_label,"SizeMRGivenAdInR_wtd","MR in R")
x_label <- str_replace(x_label,"StageDistribution_D_S1","S1")
x_label <- str_replace(x_label,"StageDistribution_D_S2","S2")
x_label <- str_replace(x_label,"StageDistribution_D_S3","S3")
x_label <- str_replace(x_label,"StageDistribution_D_S4","S4")
x_label <- str_replace(x_label,"StageDistribution_P_S1","S1")
x_label <- str_replace(x_label,"StageDistribution_P_S2","S2")
x_label <- str_replace(x_label,"StageDistribution_P_S3","S3")
x_label <- str_replace(x_label,"StageDistribution_P_S4","S4")
x_label <- str_replace(x_label,"StageDistribution_R_S1","S1")
x_label <- str_replace(x_label,"StageDistribution_R_S2","S2")
x_label <- str_replace(x_label,"StageDistribution_R_S3","S3")
x_label <- str_replace(x_label,"StageDistribution_R_S4","S4")

x_label <- str_replace(x_label,"20_39","25")
x_label <- str_replace(x_label,"30_39","35")
x_label <- str_replace(x_label,"40_49","45")
x_label <- str_replace(x_label,"50_59","55")
x_label <- str_replace(x_label,"60_69","65")
x_label <- str_replace(x_label,"70_79","75")
x_label <- str_replace(x_label,"80_99","90")



plot2 <- ggplot(data = model_outputs_targets_simcrc, 
                aes(x    = n_target, 
                    y    = targets, 
                    ymin = stopping_lower_bounds, 
                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="none") +
  # geom_line(data = model_outputs_targets,
  #           aes(x    = n_target,
  #               y    = targets,
  #               ymin = model_LB_95,
  #               ymax = model_UB_95), 
  #           color = "black") +
  geom_errorbar(data = model_outputs_targets_simcrc[model_outputs_targets_simcrc$categorical==1,],
                aes(x    = n_target-0.07,
                    y    = model_mean,
                    ymin = model_LB_95,
                    ymax = model_UB_95),width=.4, size=0.7, color="black", alpha = 0.7, position = "dodge2") +
  geom_ribbon(data = model_outputs_targets_simcrc[model_outputs_targets_simcrc$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_simcrc[model_outputs_targets_simcrc$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 3) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="none") +
  scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5))+
  scale_x_continuous(breaks= 1:lim, labels=x_label) +
  theme_bw(base_size = 23) +
  theme(plot.title = element_text(size = 22, face = "bold"),
        axis.text.x = element_text(size = 12, angle = 90),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - SimCRC", 
       x     = "", y     = "")

plot2

# load("figs/AbstractSMDM/plot_for_legend.RData")
# 
# plot2 <- ggarrange(plot2, 
#                ncol = 1, nrow = 1, legend = "bottom", legend.grob = get_legend(plot_for_legend))
# plot2

ggsave(plot2,
       filename = paste0("output/BayCANN_versions/",BayCANN_version,"/Model_validation.png"),
       width = 12, height = 12)


###### 5.2.1 Figure for abstract ----

# ====  Plot of Prevalence

model_outputs_targets_simcrc_abstract <- model_outputs_targets_simcrc[ model_outputs_targets_simcrc$target_groups%in%c("A) Prevalence 0 Ad by age"),]
model_outputs_targets_simcrc_abstract$target_groups <- "SimCRC: Prevalence of adenomas \namong women (%)"

selected_targets <- c("Prev0AdOrPreClin_age32", "Prev0AdOrPreClin_age42", "Prev0AdOrPreClin_age52",
                      "Prev0AdOrPreClin_age62", "Prev0AdOrPreClin_age72", "Prev0AdOrPreClin_age82",
                      "Prev0AdOrPreClin_age92")
model_outputs_targets_simcrc_abstract <- model_outputs_targets_simcrc_abstract[ model_outputs_targets_simcrc_abstract$target_names %in% selected_targets,]
x_label2 <- c("32" , "42", "52" , "62","72" , "82", "92" )


model_outputs_targets_simcrc_abstract$model_mean  <- (1 - model_outputs_targets_simcrc_abstract$model_mean)*100
model_outputs_targets_simcrc_abstract$targets     <- (1 - model_outputs_targets_simcrc_abstract$targets)*100
model_outputs_targets_simcrc_abstract$model_UB_95 <- (1 - model_outputs_targets_simcrc_abstract$model_UB_95)*100
model_outputs_targets_simcrc_abstract$model_LB_95 <- (1 - model_outputs_targets_simcrc_abstract$model_LB_95)*100
model_outputs_targets_simcrc_abstract$model_UB_50 <- (1 - model_outputs_targets_simcrc_abstract$model_UB_50)*100
model_outputs_targets_simcrc_abstract$model_LB_50 <- (1 - model_outputs_targets_simcrc_abstract$model_LB_50)*100
model_outputs_targets_simcrc_abstract$stopping_lower_bounds <- (1 - model_outputs_targets_simcrc_abstract$stopping_lower_bounds)*100
model_outputs_targets_simcrc_abstract$stopping_upper_bounds <- (1 - model_outputs_targets_simcrc_abstract$stopping_upper_bounds)*100

#plot_miscan_prevalence
plot_simcrc_prevalence <- ggplot(data = model_outputs_targets_simcrc_abstract, 
                                          aes(x    = n_target, 
                                              y    = targets, 
                                              ymin = stopping_lower_bounds, 
                                              ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="none") +
  geom_ribbon(data = model_outputs_targets_simcrc_abstract[model_outputs_targets_simcrc_abstract$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_simcrc_abstract[model_outputs_targets_simcrc_abstract$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 4) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="none") +
  scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(0,100))+
  scale_x_continuous(breaks= c(2,4,6,8,10,12,14), labels= x_label2) + 
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #,element_text(size = 22, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.text.y = element_text(size = 18),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - SimCRC", 
       x     = "Age", y     = "")

plot_simcrc_prevalence

save(plot_simcrc_prevalence,file = "figs/AbstractSMDM/Plot_prevalence_simcrc.RData")


# --- Plot of incidence

model_outputs_targets_simcrc_abstract <- model_outputs_targets_simcrc[ model_outputs_targets_simcrc$target_groups%in%c("J) Incidence Overall by age"),]
model_outputs_targets_simcrc_abstract$target_groups <- "SimCRC: CRC incidence \nper 100,000 women"

selected_targets <- c("Prev0AdOrPreClin_age32", "Prev0AdOrPreClin_age42", "Prev0AdOrPreClin_age52",
                      "Prev0AdOrPreClin_age62", "Prev0AdOrPreClin_age72", "Prev0AdOrPreClin_age82",
                      "Prev0AdOrPreClin_age92")
ylabel <- "Rate per 100,000 people"
#lim=500

plot_simcrc_incidence <- ggplot(data = model_outputs_targets_simcrc_abstract, 
                                 aes(x    = n_target, 
                                     y    = targets, 
                                     ymin = stopping_lower_bounds, 
                                     ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="none") +
  geom_ribbon(data = model_outputs_targets_simcrc_abstract[model_outputs_targets_simcrc_abstract$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_simcrc_abstract[model_outputs_targets_simcrc_abstract$categorical==0,],
              aes(x    = n_target,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 4) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="none") +
  scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(0,500))+
  scale_x_continuous(breaks= 1:lim, labels=x_label) + 
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #,element_text(size = 22, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.text.y = element_text(size = 18),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - SimCRC", 
       x     = "Age", y     = "")

plot_simcrc_incidence

save(plot_simcrc_incidence,file = "figs/AbstractSMDM/Plot_incidence_simcrc.RData")


#### 5.3 CRCSPIN ----

## previous work


adeno_prev_male <- c("Corley_adeno_prev_male_50_54","Corley_adeno_prev_male_55_59","Corley_adeno_prev_male_60_64",   
                     "Corley_adeno_prev_male_65_69","Corley_adeno_prev_male_70_74","Corley_adeno_prev_male_75plus")
adeno_prev_fem <- c("Corley_adeno_prev_fem_50_54","Corley_adeno_prev_fem_55_59","Corley_adeno_prev_fem_60_64",   
                     "Corley_adeno_prev_fem_65_69","Corley_adeno_prev_fem_70_74","Corley_adeno_prev_fem_75plus")

inc_colon_fem <- c("SEER_inc_colon_fem_20_49", "SEER_inc_colon_fem_50_59", "SEER_inc_colon_fem_60_69", "SEER_inc_colon_fem_70_84", "SEER_inc_colon_fem_85_100")  

inc_colon_male <- c("SEER_inc_colon_male_20_49", "SEER_inc_colon_male_50_59", "SEER_inc_colon_male_60_69", "SEER_inc_colon_male_70_84", "SEER_inc_colon_male_85_100")  

inc_rectal_fem <- c("SEER_inc_rectal_fem_20_49", "SEER_inc_rectal_fem_50_59", "SEER_inc_rectal_fem_60_69",
                    "SEER_inc_rectal_fem_70_84", "SEER_inc_rectal_fem_85_100")

inc_rectal_male <- c("SEER_inc_rectal_male_20_49", "SEER_inc_rectal_male_50_59", "SEER_inc_rectal_male_60_69",
                    "SEER_inc_rectal_male_70_84", "SEER_inc_rectal_male_85_100")

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% adeno_prev_male,
                                                      "1.a Corley adenoma prevalence (male)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% adeno_prev_fem,
                                                      "1.b Corley adenoma prevalence (female)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- str_replace(model_outputs_targets_crcspin$target_groups,"UKFSS","2. UKFSS screen detection")

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% inc_colon_male,
                                                      "3.b SEER incidence colon (male)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% inc_colon_fem,
                                                      "3.a SEER incidence colon (female)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% inc_rectal_fem,
                                                      "4.a SEER incidence rectal (female)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- ifelse(model_outputs_targets_crcspin$target_names %in% inc_rectal_male,
                                                      "4.b SEER incidence rectal (male)" ,model_outputs_targets_crcspin$target_groups)

model_outputs_targets_crcspin$target_groups <- str_replace(model_outputs_targets_crcspin$target_groups,"Pickhardt","5. Pickhardt pct adenoma")

model_outputs_targets_crcspin$target_groups <- str_replace(model_outputs_targets_crcspin$target_groups,"Church","6. Church pct crc")

model_outputs_targets_crcspin$target_groups <- str_replace(model_outputs_targets_crcspin$target_groups,"Lieberman","7. Lieberman pct crc")





# #model_outputs_targets_miscan$target_groups <- ordered(model_outputs_targets_miscan$target_groups,
#                                            levels = c("Prior",
#                                                       "Posterior IMABC_CRCSPIN",
#                                                       "Posterior BayCANN_CRCSPIN"))


#model_outputs_targets_miscan$target_names <- str_replace(model_outputs_targets_miscan$target_names,"adeno_","")

model_outputs_targets_crcspin$target_groups_num <- unclass(as.factor(model_outputs_targets_crcspin$target_groups))
model_outputs_targets_crcspin$n_target2 <- model_outputs_targets_crcspin$n_target+model_outputs_targets_crcspin$target_groups_num  
v_target_order <- model_outputs_targets_crcspin$n_target2  #vector for labels


###



#identification of categorical variables

cat_groups <- c("2. UKFSS screen detection", "5. Pickhardt pct adenoma","6. Church pct crc", "7. Lieberman pct crc")
model_outputs_targets_crcspin$categorical <- ifelse(model_outputs_targets_crcspin$target_groups %in% cat_groups,1,0)

#  X label 
string_label <- 'c("2"="Corley_adeno_prev_male_50_54", '
lim=dim(model_outputs_targets_crcspin)[1]
c=2
#for (i in 2:(lim-1)) {
for (i in v_target_order[2:(length(v_target_order)-1)]) {
  string_label <- paste(string_label,'"',i,'"=','"', model_outputs_targets_crcspin$target_names[c],'",') 
  c <- c+1
}
string_label <- paste(string_label,'"',v_target_order[length(v_target_order)],'"=','"', model_outputs_targets_crcspin$target_names[lim],'")')
x_label=eval(parse(text = string_label))

x_label <- str_replace(x_label,"Corley_adeno_prev_male_","")
x_label <- str_replace(x_label,"Corley_adeno_prev_fem_","")
x_label <- str_replace(x_label,"UKFSS_screen_det_crc_","")
x_label <- str_replace(x_label,"SEER_inc_colon_fem_","")
x_label <- str_replace(x_label,"SEER_inc_colon_male_","")
x_label <- str_replace(x_label,"SEER_inc_rectal_fem_","")
x_label <- str_replace(x_label,"SEER_inc_rectal_male_","")
x_label <- str_replace(x_label,"Pickhardt_pct_adeno_lt_","")
x_label <- str_replace(x_label,"Pickhardt_pct_adeno_6_to_lt","")
x_label <- str_replace(x_label,"Pickhardt_pct_adeno_","")

x_label <- str_replace(x_label,"Church_pct_crc_6_to_lt","")
x_label <- str_replace(x_label,"Church_pct_crc_","")
x_label <- str_replace(x_label,"Lieberman_pct_crc_6_to_lt","")
x_label <- str_replace(x_label,"Lieberman_pct_crc_","")

x_label <- str_replace(x_label,"_","-")

x_label <- str_replace(x_label,"50-54","52")
x_label <- str_replace(x_label,"55-59","57")
x_label <- str_replace(x_label,"60-64","62")
x_label <- str_replace(x_label,"65-69","67")
x_label <- str_replace(x_label,"70-74","72")
x_label <- str_replace(x_label,"75plus","75+")
x_label <- str_replace(x_label,"20-49","35")
x_label <- str_replace(x_label,"50-59","55")
x_label <- str_replace(x_label,"60-69","65")
x_label <- str_replace(x_label,"70-84","77")
x_label <- str_replace(x_label,"85-100","85+")

plot3 <- ggplot(data = model_outputs_targets_crcspin, 
                aes(x    = n_target2, 
                    y    = targets, 
                    ymin = stopping_lower_bounds, 
                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="none") +
  # geom_line(data = model_outputs_targets,
  #           aes(x    = n_target,
  #               y    = targets,
  #               ymin = model_LB_95,
  #               ymax = model_UB_95), 
  #           color = "black") +
  geom_errorbar(data = model_outputs_targets_crcspin[model_outputs_targets_crcspin$categorical==1,],
                aes(x    = n_target2-0.08,
                    y    = model_mean,
                    ymin = model_LB_95,
                    ymax = model_UB_95),width=.4, size=0.7, color="black", alpha = 0.7, position = "dodge2") +
  geom_ribbon(data = model_outputs_targets_crcspin[model_outputs_targets_crcspin$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_crcspin[model_outputs_targets_crcspin$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 3) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="none") +
  scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 18) +
  theme(plot.title = element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 90),
        axis.title = element_text(size = 15),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - CRC SPIN", 
       x     = "", y = "")

plot3

load("figs/AbstractSMDM/plot_for_legend.RData")

plot3 <- ggarrange(plot3, 
               ncol = 1, nrow = 1, legend = "bottom", legend.grob = get_legend(plot_for_legend))


ggsave(plot3,
       filename = paste0("output/BayCANN_versions/",BayCANN_version,"/Model_validation.jpg"),
       width = 13, height = 12)








###### 5.3.1 Figure for Abstract SDMD  ####


# Plot of CRC-SPIN Incidence
model_outputs_targets_crcspin_abstract <- model_outputs_targets_crcspin[ model_outputs_targets_crcspin$target_groups%in%c("3.a SEER incidence colon (female)"),]
model_outputs_targets_crcspin_abstract$target_groups <- "CRCSPIN: Incidence colon \n per 100,000 women"
#model_outputs_targets_crcspin_abstract <- model_outputs_targets_crcspin_abstract[3:8,] 
ylabel <- "Rate per 100,000 people"

model_outputs_targets_crcspin_abstract$model_mean  <- (model_outputs_targets_crcspin_abstract$model_mean)*100000
model_outputs_targets_crcspin_abstract$targets     <- (model_outputs_targets_crcspin_abstract$targets)*100000
model_outputs_targets_crcspin_abstract$model_UB_95 <- (model_outputs_targets_crcspin_abstract$model_UB_95)*100000
model_outputs_targets_crcspin_abstract$model_LB_95 <- (model_outputs_targets_crcspin_abstract$model_LB_95)*100000
model_outputs_targets_crcspin_abstract$model_UB_50 <- (model_outputs_targets_crcspin_abstract$model_UB_50)*100000
model_outputs_targets_crcspin_abstract$model_LB_50 <- (model_outputs_targets_crcspin_abstract$model_LB_50)*100000
model_outputs_targets_crcspin_abstract$stopping_lower_bounds <- (model_outputs_targets_crcspin_abstract$stopping_lower_bounds)*100000
model_outputs_targets_crcspin_abstract$stopping_upper_bounds <- (model_outputs_targets_crcspin_abstract$stopping_upper_bounds)*100000

plot_crcspin_incidence <- ggplot(data = model_outputs_targets_crcspin_abstract, 
                                aes(x    = n_target2, 
                                    y    = targets, 
                                    ymin = stopping_lower_bounds, 
                                    ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="bottom") +
  geom_ribbon(data = model_outputs_targets_crcspin_abstract[model_outputs_targets_crcspin_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_crcspin_abstract[model_outputs_targets_crcspin_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 1) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="bottom") +
  #scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5),limits = c(0,500))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - CRCSPIN", 
       x     = "Age", y = "")
#scale_color_manual( values = colors_plt )
plot_crcspin_incidence

save(plot_crcspin_incidence,file = "figs/AbstractSMDM/Plot_incidence_CRCSPIN.RData")


# Plot of CRC-SPIN Prevalence
model_outputs_targets_crcspin_abstract <- model_outputs_targets_crcspin[ model_outputs_targets_crcspin$target_groups%in%c("1.b Corley adenoma prevalence (female)"),]
model_outputs_targets_crcspin_abstract$target_groups <- "CRCSPIN: Prevalence of adenomas \n among women (%)"
ylabel <- "Prevalence"

model_outputs_targets_crcspin_abstract$targets <- model_outputs_targets_crcspin_abstract$targets*100
model_outputs_targets_crcspin_abstract$stopping_lower_bounds <- model_outputs_targets_crcspin_abstract$stopping_lower_bounds*100
model_outputs_targets_crcspin_abstract$stopping_upper_bounds <- model_outputs_targets_crcspin_abstract$stopping_upper_bounds*100
model_outputs_targets_crcspin_abstract$model_LB_95 <- model_outputs_targets_crcspin_abstract$model_LB_95*100
model_outputs_targets_crcspin_abstract$model_UB_95 <- model_outputs_targets_crcspin_abstract$model_UB_95*100
model_outputs_targets_crcspin_abstract$model_LB_50 <- model_outputs_targets_crcspin_abstract$model_LB_50*100
model_outputs_targets_crcspin_abstract$model_UB_50 <- model_outputs_targets_crcspin_abstract$model_UB_50*100


cols <- c("Target"="red","50% iqr"="black","95% iqr"="gray")

plot_crcspin_prevalence <- ggplot(data = model_outputs_targets_crcspin_abstract, 
                                 aes(x    = n_target2, 
                                     y    = targets, 
                                     ymin = stopping_lower_bounds, 
                                     ymax = stopping_upper_bounds))+ 
  geom_errorbar(width=.4, size=0.9, color="red") +
  theme(legend.position="bottom") +
  geom_ribbon(data = model_outputs_targets_crcspin_abstract[model_outputs_targets_crcspin_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_95,
                  ymax = model_UB_95),
              fill = "black",
              alpha = 0.3) +
  geom_ribbon(data = model_outputs_targets_crcspin_abstract[model_outputs_targets_crcspin_abstract$categorical==0,],
              aes(x    = n_target2,
                  y    = targets,
                  ymin = model_LB_50,
                  ymax = model_UB_50),
              fill = "black",
              alpha = 0.5) +
  facet_wrap(~ target_groups,scales="free", ncol = 1) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(), legend.position="bottom") +
  #scale_fill_manual(values = c("grey10", "grey30"))+
  scale_y_continuous(breaks = number_ticks(5), limits = c(0,100))+
  # scale_x_continuous(breaks= 1:lim, labels=x_label) +
  scale_x_continuous(breaks= v_target_order, labels=x_label) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_blank(), #element_text(size = 23, face = "bold"),
        axis.text.x = element_text(size = 18, angle = 0),
        axis.title = element_text(size = 18),
        panel.grid.major = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0)) +
  labs(title = "BayCANN - CRCSPIN", 
       x     = "Age", y = "")
#scale_color_manual( values = colors_plt )

plot_crcspin_prevalence
save(plot_crcspin_prevalence,file = "figs/AbstractSMDM/Plot_prevalence_CRCSPIN.RData")







