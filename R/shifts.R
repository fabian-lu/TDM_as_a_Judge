# Where does the variation in the probability shift come from? Linear mixed model with crossed random
# effects, used only to split the variance (variance components), not to test anything.
#   Rscript R/shifts.R           (from the repository root, after explore.save())
suppressPackageStartupMessages({library(data.table); library(lme4)})
d <- fread(cmd = "zcat data/exploratory/long.csv.gz")
d <- d[bias != "fallacy" & !is.na(shift)]

split_variance <- function(formula, data, label) {
  t0 <- Sys.time()
  fit <- lmer(formula, data = data, REML = TRUE,
              control = lmerControl(calc.derivs = FALSE))
  vc <- as.data.frame(VarCorr(fit))[, c("grp", "vcov")]
  vc$share <- round(vc$vcov / sum(vc$vcov), 3)
  vc$vcov <- signif(vc$vcov, 3)
  cat(sprintf("\n== %s (n = %d, %.0f s)\n", label, nrow(data), as.numeric(Sys.time() - t0, units = "secs")))
  print(vc[order(-vc$share), ], row.names = FALSE)
  vc
}

f <- shift ~ 1 + (1 | bias) + (1 | model) + (1 | format) + (1 | bias:model) + (1 | bias:format) +
  (1 | model:format) + (1 | source) + (1 | item) + (1 | item:bias) + (1 | item:model)
v1 <- split_variance(f, d, "signed shift, all formats")
d[, abs_shift := abs(shift)]
v2 <- split_variance(update(f, abs_shift ~ .), d, "size of the shift |p_hat - p|, all formats")

# estimates for the paper's appendix tables (07_figures)
fwrite(rbind(data.table(outcome = "signed", v1), data.table(outcome = "size", v2)), "data/exploratory/variance_shift.csv")
