# What makes a verdict flip? Logistic mixed model with a random effect per item.
#   Rscript R/flips.R            (from the repository root, after explore.save() has written data/exploratory/long.csv.gz)
# One line per model, format and perturbed prompt; fallacy oversight is left out (there a flip is correct).
suppressPackageStartupMessages({library(data.table); library(lme4)})
d <- fread(cmd = "zcat data/exploratory/long.csv.gz")
d <- d[bias != "fallacy" & !is.na(flip)]
d[, `:=`(bias = relevel(factor(bias), "diversity, neutral note"), model = relevel(factor(model), "jev-1.13"),
         format = relevel(factor(format), "calm"), source = factor(source),
         margin10 = margin * 10)]                 # margin in steps of 0.1

odds <- function(fit) {
  b <- fixef(fit); se <- sqrt(diag(vcov(fit)))
  round(data.frame(odds_ratio = exp(b), lo = exp(b - 1.96 * se), hi = exp(b + 1.96 * se)), 3)
}

cat("== 1. CALM format: flip ~ margin + original correct + length ratio + bias + model + source + (1 | item)\n")
c <- d[format == "calm"]
t0 <- Sys.time()
m1 <- glmer(flip ~ margin10 + orig_correct + log_len_ratio + bias + model + source + (1 | item),
            data = c, family = binomial, nAGQ = 0)
o1 <- odds(m1)
print(o1)
v <- as.numeric(VarCorr(m1)$item)
cat(sprintf("item variance %.3f (logit scale), ICC %.3f, n = %d prompts, %d items, %.0f s\n",
            v, v / (v + pi^2 / 3), nrow(c), length(unique(c$item)), as.numeric(Sys.time() - t0, units = "secs")))

cat("\n== 2. Does the margin effect differ between models? (CALM format, margin x model)\n")
m2 <- glmer(flip ~ margin10 * model + bias + (1 | item), data = c, family = binomial, nAGQ = 0)
b <- fixef(m2); k <- grep("^margin10", names(b))
print(round(exp(c(b["margin10"], b["margin10"] + b[k[-1]])), 3))
cat("(odds ratio per 0.1 of margin; first value = Jev, the reference)\n")

cat("\n== 3. All formats: flip ~ margin + bias + model + format + (1 | item) + (1 | item:model)\n")
t0 <- Sys.time()
m3 <- glmer(flip ~ margin10 + bias + model + format + (1 | item) + (1 | item:model),
            data = d, family = binomial, nAGQ = 0)
print(odds(m3)[c("margin10", "formatnative", "formatdecomposed"), ])
vc <- as.data.frame(VarCorr(m3))
print(vc[, c("grp", "vcov")])
cat(sprintf("n = %d, %.0f s\n", nrow(d), as.numeric(Sys.time() - t0, units = "secs")))

# estimates for the paper's appendix tables (07_figures)
o3 <- odds(m3)[c("formatnative", "formatdecomposed"), ]
est <- rbind(o1, o3)
fwrite(data.table(term = rownames(est), est), "data/exploratory/flips_odds.csv")
fwrite(data.table(n = nrow(c), items = length(unique(c$item)), item_variance = v, icc = v / (v + pi^2 / 3)),
       "data/exploratory/flips_info.csv")
