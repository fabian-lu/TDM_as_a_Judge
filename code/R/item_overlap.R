# Do TDMs and LLM judges flip on the same prompts? Logistic mixed model with two random effects per prompt.
#   Rscript R/item_overlap.R      (from code/, after 09_llm_analysis section 14 has written
#                                   outputs/exploratory/flips_judges.csv.gz; the models run in parallel)
# One line per judge and perturbed prompt: the eight TDMs in the CALM format and the five 2026 LLM judges, eleven
# tests (fallacy oversight is left out, there a flip is correct). flip = 1 if the verdict on the perturbed prompt
# differs from that on the original.
#
# Fixed effects: one intercept per judge and test (judge:bias), so that the random effects describe only how a
# prompt departs from the flip rate of that judge in that test. Random effects: per prompt, one value for each of two
# groups of judges (A and B), with their correlation. A correlation near 1 means the prompts that make the judges of
# group A flip are the ones that make those of group B flip. Judge-test cells without any flip (or with only flips)
# carry no information on the prompts and are left out.
#
#   1. TDMs against LLM judges, all tests
#   2. Benchmark: the same correlation between two halves of the TDMs, two halves of the LLM judges, and a half of
#      each kind (five random splits), so that a correlation across kinds can be compared with one within a kind
#   3. TDMs against LLM judges, each test on its own
#   4. TDMs against LLM judges with a second level for the item (several prompts share an item)
suppressPackageStartupMessages({library(data.table); library(lme4); library(parallel)})
d <- fread(cmd = "zcat outputs/exploratory/flips_judges.csv.gz")
d[, cell := paste(judge, bias, sep = " | ")]
d <- d[, if (sum(flip) > 0 & sum(flip) < .N) .SD, by = cell]
d[, cell := factor(cell)]
cat(sprintf("%d lines, %d prompts, %d items, %d judge-test cells\n",
            nrow(d), uniqueN(d$id), uniqueN(d$item), uniqueN(d$cell)))

latent <- pi^2 / 3            # variance of the logistic residual
fit_pair <- function(data, a, b, item_level = FALSE) {
  # a, b: judges of group A and B; returns the SDs of the prompt effects, their correlation and the latent ICCs
  e <- data[judge %in% c(a, b)]
  e[, `:=`(A = as.numeric(judge %in% a), B = as.numeric(judge %in% b))]
  e[, cell := droplevels(cell)]
  f <- if (item_level) flip ~ 0 + cell + (0 + A + B | item) + (0 + A + B | id) else flip ~ 0 + cell + (0 + A + B | id)
  t0 <- Sys.time()
  m <- glmer(f, data = e, family = binomial, nAGQ = 0, control = glmerControl(calc.derivs = FALSE))
  out <- function(g) {
    vc <- VarCorr(m)[[g]]; sd <- attr(vc, "stddev"); r <- attr(vc, "correlation")
    data.table(level = g, sd_A = sd[["A"]], sd_B = sd[["B"]], cor = r["A", "B"],
               icc_A = sd[["A"]]^2 / (sd[["A"]]^2 + latent), icc_B = sd[["B"]]^2 / (sd[["B"]]^2 + latent))
  }
  res <- rbindlist(lapply(if (item_level) c("item", "id") else "id", out))
  res[, `:=`(n = nrow(e), prompts = uniqueN(e$id), singular = isSingular(m),
             seconds = round(as.numeric(Sys.time() - t0, units = "secs")))]
  list(res = res, model = if (!item_level && length(a) == 8 && length(b) == 5 && uniqueN(e$bias) == 11) m else NULL)
}

TDMS <- sort(unique(d[kind == "TDM"]$judge)); LLMS <- sort(unique(d[kind == "LLM"]$judge))
jobs <- list(list(name = "TDM - LLM, all tests", data = "all", a = TDMS, b = LLMS, item = FALSE),
             list(name = "TDM - LLM, all tests, item and prompt", data = "all", a = TDMS, b = LLMS, item = TRUE))
set.seed(20261011)
for (s in 1:5) {
  ta <- sample(TDMS, 4); tb <- setdiff(TDMS, ta); la <- sample(LLMS, 3); lb <- setdiff(LLMS, la)
  add <- function(kind, a, b) list(name = sprintf("split %d, %s", s, kind), data = "all", a = a, b = b, item = FALSE)
  jobs <- c(jobs, list(add("TDM - TDM", ta, tb), add("LLM - LLM", la, lb), add("TDM - LLM", ta, la),
                       add("TDM - LLM", ta, lb), add("TDM - LLM", tb, la), add("TDM - LLM", tb, lb)))
}
for (b in sort(unique(d$bias))) jobs <- c(jobs, list(list(name = paste("test:", b), data = b, a = TDMS, b = LLMS,
                                                          item = FALSE)))
cat(length(jobs), "models\n")
t0 <- Sys.time()
done <- mclapply(jobs, function(j) {
  r <- fit_pair(if (j$data == "all") d else d[bias == j$data], j$a, j$b, j$item)
  r$res[, model := j$name]; r
}, mc.cores = 7, mc.preschedule = FALSE)
cat(sprintf("all models fitted in %.0f min\n", as.numeric(Sys.time() - t0, units = "mins")))
res <- rbindlist(lapply(done, function(r) if (is.list(r)) r$res else data.table(model = "failed")), fill = TRUE)
setcolorder(res, "model")
print(res, digits = 3, nrows = 200)

cat("\n== Benchmark: mean correlation over the five splits\n")
res[grepl("^split", model), pair := sub("^split [0-9], ", "", model)]
print(res[!is.na(pair), .(cor = mean(cor), min = min(cor), max = max(cor), models = .N), by = pair], digits = 3)
fwrite(res, "outputs/exploratory/item_overlap_model.csv")
m1 <- Filter(Negate(is.null), lapply(done, function(r) if (is.list(r)) r$model))
if (length(m1)) saveRDS(m1[[1]], "outputs/exploratory/item_overlap_m1.rds")    # for the profile interval
