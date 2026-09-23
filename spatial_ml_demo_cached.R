################################################################################
##  Spatial ML for Spatial Regression -- Software Demonstration (California housing)
##
##  REAL-TIME DEMO WORKFLOW
##  ----------------------------------------------------------------------------
##  The only slow methods are RF-GLS and the whitening transform. They are
##  computed ONCE and cached to disk; everything else runs live in seconds.
##
##    * BEFORE the talk : set  refit_slow <- TRUE , run the whole script once.
##                        (builds "slow_methods_cache.rds"; RF-GLS may take a while)
##    * DURING the talk : set  refit_slow <- FALSE , run again -> instant.
##
##  The cache is tied to the random seed and the n_train_max/n_test_max below.
##  If you change those, rebuild the cache (refit_slow <- TRUE).
################################################################################
rm(list = ls())

refit_slow <- TRUE                      # <- TRUE to (re)build cache, FALSE for live
cache_file <- "slow_methods_cache.rds"

## ---- packages -------------------------------------------------------------
library(dplyr); library(tidyr); library(ggplot2)
library(ranger)              # plain random forest
library(GpGp)                # Vecchia GP (benchmark) -- fast, runs live
library(BRISC)               # NNGP fit (shared spatial params) -- only in precompute
suppressWarnings({ library(fields); library(spNNGP); library(magrittr); library(parallel) })
library(LatticeKrig)
library(GPvecchia)
library(fields)
library(geoR)
set.seed(12345)
cores <- max(1L, parallel::detectCores() - 1L)

## ---------------------------------------------------------------------------
## 1.  READ + PREPROCESS  (from lec1_ibc.R)
## ---------------------------------------------------------------------------
house <- read.csv("../data/housing.csv")            # <-- adjust path
house <- na.omit(house) %>% dplyr::filter(median_house_value < 500000)
house$ocean_proximity <- as.factor(house$ocean_proximity)
dim(house)
colnames(house)

# Assuming we have longitude and latitude columns

ggplot2::map_data("state") %>%
  filter(region == "california") -> california_map
ggplot() +
  geom_polygon(data = california_map, aes(x = long, y = lat, group = group),
               fill = "lightgray", color = "black") +
  geom_point(data = house, aes(x = longitude, y = latitude, color = median_house_value),
             alpha = 0.6, size = 2) +
  scale_color_viridis_c(option = "plasma", name = "Price (USD)") +
  labs(title = "Housing Data in California",
       x = "Longitude", y = "Latitude") +
  theme_minimal()

## Non-spatial OLS baseline (exactly as in lec1_ibc.R)
ols_full <- lm(median_house_value ~ median_income + housing_median_age +
                 total_rooms + total_bedrooms + population + households +
                 ocean_proximity, data = house)
cat("OLS R^2 (full data):", round(summary(ols_full)$r.squared, 3), "\n")

house$res=house$median_house_value-unname(ols_full$fitted.values)

ggplot() +
  geom_polygon(data = california_map, aes(x = long, y = lat, group = group),
               fill = "lightgray", color = "black") +
  geom_point(data = house, aes(x = longitude, y = latitude, color = res),
             alpha = 0.8, size = 2) +
  scale_color_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "Price") +
  labs(title = "Housing Data in California",
       x = "Longitude", y = "Latitude") +
  theme_minimal()


# Convert the data to a geoR object
geo_data <- as.geodata(house %>% mutate(latitude=jitter(latitude)),
                       coords.col = c("longitude", "latitude"), data.col = "res")

# Create a variogram for the house prices using geoR
variogram_price <- variog(geo_data)

# Plot the variogram
plot(variogram_price, main = "Variogram of House Prices")
## ---------------------------------------------------------------------------
## 2.  DESIGN MATRIX + COORDINATES (built once; same X for every method)
## ---------------------------------------------------------------------------
mm     <- model.matrix(~ median_income + housing_median_age + total_rooms +
                         total_bedrooms + population + households + ocean_proximity,
                       data = house)
Xnoint <- mm[, -1, drop = FALSE]
y      <- house$median_house_value / 1000           # $1000s
locs   <- as.matrix(house[, c("longitude", "latitude")])
set.seed(1)                                          # break exact-duplicate coords
locs <- locs + matrix(rnorm(length(locs), sd = 1e-3), ncol = 2)

## ---------------------------------------------------------------------------
## 3.  SPATIAL BLOCK HOLD-OUT (fixed by the seeds above -> reproducible split)
## ---------------------------------------------------------------------------
nb  <- 10
bx  <- cut(locs[,1], breaks = nb, labels = FALSE)
by  <- cut(locs[,2], breaks = nb, labels = FALSE)
blk <- interaction(bx, by, drop = TRUE)
test_blocks <- sample(levels(blk), size = ceiling(0.20 * nlevels(blk)))
is_test     <- blk %in% test_blocks

## RF-GLS scales super-linearly in n and is the binding constraint: n=3000 can
## take HOURS. Keep training modest so all methods stay comparable AND tractable.
n_train_max <- 3000                                 # RF-GLS-friendly; bump only if you must
n_test_max  <- 2000
tr_idx <- which(!is_test); te_idx <- which(is_test)
if (length(tr_idx) > n_train_max) tr_idx <- sample(tr_idx, n_train_max)
if (length(te_idx) > n_test_max)  te_idx <- sample(te_idx, n_test_max)

Xtr <- Xnoint[tr_idx, , drop = FALSE]; Xte <- Xnoint[te_idx, , drop = FALSE]
ytr <- y[tr_idx];                      yte <- y[te_idx]
ltr <- locs[tr_idx, , drop = FALSE];   lte <- locs[te_idx, , drop = FALSE]
keep <- apply(Xtr, 2, function(z) length(unique(z)) > 1)
Xtr  <- Xtr[, keep, drop = FALSE]; Xte <- Xte[, keep, drop = FALSE]
mtry_rf <- max(1, floor(ncol(Xtr) / 3))
cat(sprintf("Train n = %d, Test n = %d\n", length(ytr), length(yte)))

rmse <- function(o, p) sqrt(mean((o - p)^2))
r2   <- function(o, p) 1 - mean((o - p)^2) / var(o)
results <- list()
add_res <- function(name, spatial, nonlinear, pred, secs)
  results[[name]] <<- c(spatial = spatial, nonlinear = nonlinear,
                        RMSE = rmse(yte, pred), R2 = r2(yte, pred),
                        Time_s = as.numeric(secs))

## ===========================================================================
## FAST METHODS -- always run live
## ===========================================================================
## 4. OLS
tr_df <- data.frame(y = ytr, Xtr); te_df <- data.frame(Xte)
t_ols <- system.time({ ols <- lm(y ~ ., data = tr_df); p_ols <- predict(ols, te_df) })["elapsed"]
add_res("OLS", 0, 0, p_ols, t_ols)

## 5. Random forest
t_rf <- system.time({
  rf   <- ranger(y ~ ., data = tr_df, num.trees = 500, mtry = mtry_rf,
                 min.node.size = 10, num.threads = cores)
  p_rf <- predict(rf, data = te_df)$predictions
})["elapsed"]
add_res("RF", 0, 1, p_rf, t_rf)

## 6. GpGp -- Vecchia GP benchmark
Xtr_gp <- cbind(1, as.matrix(Xtr)); Xte_gp <- cbind(1, as.matrix(Xte))
t_gp <- system.time({
  gp_fit <- fit_model(y = ytr, locs = ltr, X = Xtr_gp, covfun_name = "matern_isotropic")
  p_gp   <- predictions(fit = gp_fit, locs_pred = lte, X_pred = Xte_gp,
                        y_obs = ytr, locs_obs = ltr, X_obs = Xtr_gp)
})["elapsed"]
add_res("GpGp", 1, 0, p_gp, t_gp)

## ===========================================================================
## SLOW METHODS -- precompute once, then load from cache
## ===========================================================================
if (refit_slow || !file.exists(cache_file)) {

  library(RandomForestsGLS)
  source("TransformFunctions_win.R")

  ## shared spatial covariance (exponential = Matern nu=1/2)
  br0 <- BRISC_estimation(coords = ltr, y = ytr, x = cbind(1, as.matrix(Xtr)),
                          cov.model = "exponential", n.neighbors = 15, verbose = FALSE)
  th  <- br0$Theta
  rng <- as.numeric(1 / th["phi"])
  nugp<- as.numeric(th["tau.sq"] / (th["sigma.sq"] + th["tau.sq"]))

  ## 7. Whitening + RF
  wtr <- data.frame(y = ytr, Xtr); wte <- data.frame(Xte); colnames(wte) <- colnames(wtr)[-1]
  t_wh <- system.time({
    wobj <- transform_to_ind(formula = y ~ ., trainData = wtr, trainLocs = ltr,
                             testData = wte, testLocs = lte,
                             MaternParams = c(rng, nugp), smoothness = 1/2,
                             M = 30, ncores = cores)
    rf_wh    <- ranger(y ~ ., data = wobj$trainData, num.trees = 500,
                       mtry = mtry_rf, min.node.size = 10, num.threads = cores)
    p_wh_ind <- predict(rf_wh, data = wobj$testData)$predictions
    p_wh     <- as.numeric(back_transform_to_spatial(p_wh_ind, wobj))
  })["elapsed"]

  ## 8. RF-GLS (fixed params + shallow trees + REAL parallelism for speed)
  ##    KEY: h = cores parallelizes the trees (default h=1 = single-threaded, the
  ##    reason it ran for hours). n_omp threads the NNGP algebra. Large nthsize
  ##    keeps trees shallow. Fixed sigma/tau/phi skips re-estimation.
  t_rfgls <- system.time({
    rfgls <- RFGLS_estimate_spatial(coords = ltr, y = ytr, X = as.matrix(Xtr),
                                    ntree = 25, mtry = mtry_rf, nthsize = 100,
                                    cov.model = "exponential", n.neighbors = 10,
                                    h = cores, n_omp = cores,   # <- h = tree parallelism
                                    param_estimate = FALSE,
                                    sigma.sq = as.numeric(th["sigma.sq"]),
                                    tau.sq   = as.numeric(th["tau.sq"]),
                                    phi      = as.numeric(th["phi"]))
    p_rfgls <- RFGLS_predict_spatial(rfgls, coords.0 = lte, Xtest = as.matrix(Xte))$prediction
  })["elapsed"]

  saveRDS(list(p_wh = p_wh, t_wh = t_wh, p_rfgls = p_rfgls, t_rfgls = t_rfgls,
               yte = yte), cache_file)
  cat(sprintf("Cached slow methods -> %s  (whitening %.1fs, RF-GLS %.1fs)\n",
              cache_file, t_wh, t_rfgls))

} else {
  cc <- readRDS(cache_file)
  ## guard: cache must match the current split
  if (length(cc$yte) != length(yte) || max(abs(cc$yte - yte)) > 1e-6)
    stop("Cache does not match current data/split. Re-run with refit_slow <- TRUE.")
  p_wh <- cc$p_wh; t_wh <- cc$t_wh; p_rfgls <- cc$p_rfgls; t_rfgls <- cc$t_rfgls
  cat("Loaded slow-method results from cache (instant).\n")
}
add_res("Whitening_RF", 1, 1, p_wh,    t_wh)
add_res("RF_GLS",       1, 1, p_rfgls, t_rfgls)

## ---------------------------------------------------------------------------
## 9.  RESULTS TABLE (accuracy + runtime; runtimes are the TRUE offline costs)
## ---------------------------------------------------------------------------
tab <- as.data.frame(do.call(rbind, results)); tab$Method <- rownames(tab)
tab <- tab[order(tab$RMSE), c("Method","spatial","nonlinear","RMSE","R2","Time_s")]
tab$RMSE <- round(tab$RMSE, 1); tab$R2 <- round(tab$R2, 3); tab$Time_s <- round(tab$Time_s, 3)
cat("\n===== Spatial hold-out accuracy & runtime (RMSE in $1000s) =====\n")
print(tab, row.names = FALSE)
write.csv(tab, "spatial_ml_results.csv", row.names = FALSE)

## ===========================================================================
## 10.  COMPARISON PLOTS
## ===========================================================================
ca <- california_map
method_levels <- c("OLS","RF","GpGp","Whitening_RF","RF_GLS")
pred_df <- data.frame(lon = lte[,1], lat = lte[,2], Truth = yte,
                      OLS = p_ols, RF = p_rf, GpGp = p_gp,
                      Whitening_RF = p_wh, RF_GLS = p_rfgls)

## 10a. prediction maps (truth + all methods, shared scale)
map_long <- pred_df %>%
  pivot_longer(all_of(c("Truth", method_levels)), names_to = "Method", values_to = "Value") %>%
  mutate(Method = factor(Method, levels = c("Truth", method_levels)))
p_maps <- ggplot() +
  geom_polygon(data = ca, aes(long, lat, group = group),
               fill = "grey93", colour = "grey70", linewidth = 0.2) +
  geom_point(data = map_long, aes(lon, lat, colour = Value), size = 0.6) +
  scale_colour_viridis_c(option = "plasma", name = "$1000s") +
  coord_quickmap() + facet_wrap(~ Method, ncol = 3) +
  labs(title = "Test-set truth and spatial predictions", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(axis.text = element_blank(), panel.grid = element_blank())
ggsave("fig_prediction_maps.png", p_maps, width = 11, height = 6.5, dpi = 150)

## 10b. error maps (pred - truth)
err_long <- pred_df %>%
  pivot_longer(all_of(method_levels), names_to = "Method", values_to = "Pred") %>%
  mutate(Method = factor(Method, levels = method_levels), Error = Pred - Truth)
p_err <- ggplot() +
  geom_polygon(data = ca, aes(long, lat, group = group),
               fill = "grey93", colour = "grey70", linewidth = 0.2) +
  geom_point(data = err_long, aes(lon, lat, colour = Error), size = 0.6) +
  scale_colour_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "Pred - Obs") +
  coord_quickmap() + facet_wrap(~ Method, ncol = 3) +
  labs(title = "Prediction error in space (red = under, blue = over)", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) + theme(axis.text = element_blank(), panel.grid = element_blank())
ggsave("fig_error_maps.png", p_err, width = 11, height = 4.5, dpi = 150)

## 10c. observed vs predicted, faceted, with per-method RMSE
sc_long <- pred_df %>%
  pivot_longer(all_of(method_levels), names_to = "Method", values_to = "Pred") %>%
  mutate(Method = factor(Method, levels = method_levels))
lab_df <- sc_long %>% group_by(Method) %>%
  summarise(RMSE = sqrt(mean((Truth - Pred)^2)), .groups = "drop") %>%
  mutate(x = min(pred_df$Truth), yv = max(sc_long$Pred))
p_scatter <- ggplot(sc_long, aes(Truth, Pred)) +
  geom_point(alpha = 0.25, size = 0.6, colour = "grey30") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = 2) +
  geom_text(data = lab_df, aes(x = x, y = yv, label = paste0("RMSE = ", round(RMSE,1))),
            hjust = 0, vjust = 1, size = 3, colour = "blue") +
  facet_wrap(~ Method, ncol = 3) +
  labs(title = "Observed vs predicted (test set)",
       x = "Observed ($1000s)", y = "Predicted ($1000s)") +
  theme_minimal(base_size = 10) + theme(aspect.ratio = 1)
ggsave("fig_obs_vs_pred.png", p_scatter, width = 10, height = 6.5, dpi = 150)

## 10d. RMSE and runtime bars
p_rmse_bar <- ggplot(tab, aes(reorder(Method, -RMSE), RMSE)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "Spatial hold-out RMSE", x = NULL, y = "RMSE ($1000s)") +
  theme_minimal(base_size = 11)
ggsave("fig_rmse_bar.png", p_rmse_bar, width = 6, height = 3.5, dpi = 150)
p_time_bar <- ggplot(tab, aes(reorder(Method, Time_s), Time_s)) +
  geom_col(fill = "darkorange") + coord_flip() +
  geom_text(aes(label = paste0(round(Time_s,1), "s")), hjust = -0.1, size = 3) +
  expand_limits(y = max(tab$Time_s) * 1.15) +
  labs(title = "Runtime by method", x = NULL, y = "seconds") +
  theme_minimal(base_size = 11)
ggsave("fig_runtime_bar.png", p_time_bar, width = 6, height = 3.5, dpi = 150)

print(p_maps); print(p_scatter); print(p_rmse_bar); print(p_time_bar); print(p_err)
################################################################################
