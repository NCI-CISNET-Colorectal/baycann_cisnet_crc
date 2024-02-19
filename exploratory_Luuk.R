library("PerformanceAnalytics")
library("ggplot2")
library("tidyr")
library("corrplot")
library("reshape2")

.libPaths(c("C:/Users/057111/RlibCustom/4.2", "P:/R/R-4.2.1/library"))
setwd("C:/Users/057111/PycharmProjects/baycann_cisnet_crc")

# Load parameters and targets
pars <- read.csv("data-raw/20240205_MISCANColon_LHS/SimulatedParameters_20240205_1322PST.csv")
tars <- read.csv("data-raw/20240205_MISCANColon_LHS/SimulatedTargets_20240205_1322PST.csv")

# Select parameter and targets
#tars_selected <- tars[c("inc_us_60_69", "int_canc_r2")]
tars_selected <- tars[, !names(tars) %in% c("iter", "draw", "step")]
#pars_selected <- pars[, names(pars) %in% c("age_factors_k", "age_factors_v", "age_factors_M", "age_factors_L")]
pars_selected <- pars[, !names(pars) %in% c("iter", "draw", "step", "seed", "sample_wt")]

# Generate correlation between parameters and targets
#sort(cor(pars_selected, tars_selected)[,2])
corr_wide = cor(pars_selected, tars_selected)

# Plot heatmap
png("data-raw/20240205_MISCANColon_LHS/corplot.png")
heatmap(corr_wide, 
        col = colorRampPalette(c("blue", "white", "red"))(20), # Choose a color palette
        #main = "Correlation Heatmap between Variables in df1 and df2",
        #xlab = "Variables in df1", ylab = "Variables in df2",
        cexRow = 1.2, cexCol = 1.2, margins = c(10, 10),
        scale="column", Colv = NA, Rowv = NA,
        symm=T, width=3, height=2)
legend("bottomright", legend = c("-1", "0", "1"), fill = colorRampPalette(c("blue", "white", "red"))(3),
       title = "Correlation", bty = "n", cex = 0.8)
dev.off()


pars_selected <- pars[, names(pars) %in% c("hazard_variance", "hazard_mean_uk", "hazard_mean_nl", 
                                           "age_factors_k", "age_factors_v", "age_factors_M", "age_factors_L",
                                           "non_progr_0", "non_progr_45", "non_progr_65", "non_progr_100"
                                           #"small_progr", "small_non_progr", 
                                           #"dwell_pcl_shape", "dwell_pcl3"
)]
panel.hist <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  his <- hist(x, plot = FALSE)
  breaks <- his$breaks
  nB <- length(breaks)
  y <- his$counts
  y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = rgb(0, 1, 1, alpha = 0.5), ...)
  # lines(density(x), col = 2, lwd = 2) # Uncomment to add density lines
}
pairs(pars_selected, upper.panel = NULL, diag.panel = panel.hist)
