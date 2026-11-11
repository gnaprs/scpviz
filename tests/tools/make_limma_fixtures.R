# Generates reference outputs from R limma for validating scpviz's moderated t-test.
# Dev/test-only (R is NOT a runtime dependency). Re-run only to regenerate fixtures:
#   Rscript tests/tools/make_limma_fixtures.R
# Model: y = intercept + beta * I(group == g1); compare g1 - g2 (coefficient 2).
suppressMessages(library(limma))
out_dir <- file.path("tests", "data", "limma")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20261111)

# Simulate log2 abundances: gene-wise variances from a scaled inverse chi-square prior
sim <- function(n_prot, n1, n2, s0sq = 0.1, d0 = 4, frac_de = 0.1, lfc = 1) {
  n <- n1 + n2
  s2 <- s0sq * d0 / rchisq(n_prot, df = d0)
  base <- rnorm(n_prot, 20, 2)
  e <- matrix(rnorm(n_prot * n), n_prot, n) * sqrt(s2) + base
  de <- seq_len(round(n_prot * frac_de))
  sgn <- sample(c(-1, 1), length(de), replace = TRUE)
  e[de, seq_len(n1)] <- e[de, seq_len(n1)] + lfc * sgn
  dimnames(e) <- list(sprintf("P%04d", seq_len(n_prot)), c(sprintf("a%d", seq_len(n1)), sprintf("b%d", seq_len(n2))))
  list(expr = e, group = c(rep("g1", n1), rep("g2", n2)))
}

save_case <- function(name, expr, group) {
  write.csv(expr, file.path(out_dir, paste0(name, "_expr.csv")), na = "NA")
  write.csv(data.frame(sample = colnames(expr), group = group), file.path(out_dir, paste0(name, "_group.csv")), row.names = FALSE)
  g <- factor(group, levels = c("g2", "g1"))
  design <- model.matrix(~g)
  for (leg in c("NULL", "TRUE", "FALSE")) {
    legacy <- switch(leg, "NULL" = NULL, "TRUE" = TRUE, "FALSE" = FALSE)
    tag <- paste0(name, "_legacy", leg)
    res <- tryCatch({
      fit <- lmFit(expr, design)
      fit <- eBayes(fit, trend = FALSE, robust = FALSE, legacy = legacy)
      list(ok = TRUE, fit = fit)
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)),
       warning = function(w) {
         # re-run suppressing warnings so numbers are still produced; record the warning text
         fit <- suppressWarnings(eBayes(lmFit(expr, design), trend = FALSE, robust = FALSE, legacy = legacy))
         list(ok = TRUE, fit = fit, warn = conditionMessage(w))
       })
    if (!res$ok) {
      write.csv(data.frame(status = "error", message = res$msg), file.path(out_dir, paste0(tag, "_prior.csv")), row.names = FALSE)
      next
    }
    fit <- res$fit
    sd_unscaled <- fit$stdev.unscaled[, 2]
    prior <- data.frame(status = "ok", s2_prior = fit$s2.prior[1], df_prior = fit$df.prior[1],
                        df_pooled = sum(fit$df.residual, na.rm = TRUE),
                        warning = if (is.null(res$warn)) "" else res$warn,
                        limma_version = as.character(packageVersion("limma")), r_version = R.version.string)
    if (length(fit$s2.prior) > 1) prior$s2_prior <- NA
    if (length(fit$df.prior) > 1) prior$df_prior <- NA
    write.csv(prior, file.path(out_dir, paste0(tag, "_prior.csv")), row.names = FALSE)
    resdf <- data.frame(protein = rownames(expr), coef = fit$coefficients[, 2], stdev_unscaled = sd_unscaled,
                        s2 = fit$sigma^2, df_residual = fit$df.residual, s2_post = fit$s2.post,
                        t = fit$t[, 2], df_total = fit$df.total, p_value = fit$p.value[, 2])
    resdf$p_adj <- p.adjust(resdf$p_value, "BH")
    write.csv(resdf, file.path(out_dir, paste0(tag, "_result.csv")), row.names = FALSE, na = "NA")
  }
}

# 1. balanced 4 v 4, complete
s <- sim(400, 4, 4); save_case("balanced_4v4", s$expr, s$group)
# 2. unbalanced 3 v 5, complete
s <- sim(400, 3, 5); save_case("unbalanced_3v5", s$expr, s$group)
# 3. mixed missingness -> variable df (ensure >=2 per group so contrast always estimable)
s <- sim(500, 4, 4); e <- s$expr
for (i in seq_len(nrow(e))) { k <- rbinom(8, 1, 0.25) == 1; if (sum(!k[1:4]) >= 2 && sum(!k[5:8]) >= 2) e[i, k] <- NA }
save_case("missing_mixed_df", e, s$group)
# 4. extreme missingness: 1v1 (df 0), one group fully missing, 1v3, all-NA row
s <- sim(300, 4, 4); e <- s$expr
for (i in 21:300) { k <- rbinom(8, 1, 0.2) == 1; if (sum(!k[1:4]) >= 2 && sum(!k[5:8]) >= 2) e[i, k] <- NA }
e[1, c(2:4, 6:8)] <- NA        # 1 v 1  -> df.residual 0
e[2, 1:4] <- NA                # group1 entirely missing
e[3, 5:8] <- NA                # group2 entirely missing
e[4, c(2:4)] <- NA; e[4, 5] <- NA   # 1 v 3
e[5, ] <- NA                   # all NA
e[6, c(2:4, 6:8)] <- NA; e[7, c(2:4, 6:8)] <- NA   # more 1 v 1
save_case("missing_extreme", e, s$group)
# 5. zero / near-zero variance, wide dynamic range (complete, equal df)
s <- sim(300, 4, 4); e <- s$expr
e[1:5, 1:4] <- matrix(rep(e[1:5, 1], 4), 5, 4); e[1:5, 5:8] <- matrix(rep(e[1:5, 5], 4), 5, 4)   # exact zero variance
e[6:10, ] <- e[6:10, ] * 1e-6 + 10                                                        # tiny variances
e[11:15, ] <- e[11:15, ] * 50                                                             # very large variances
save_case("zero_variance", e, s$group)
# 5b. zero variance + missingness (unequal df path with zero variances)
e2 <- e
for (i in 16:300) { k <- rbinom(8, 1, 0.2) == 1; if (sum(!k[1:4]) >= 2 && sum(!k[5:8]) >= 2) e2[i, k] <- NA }
save_case("zero_variance_missing", e2, s$group)
# 6. no differential expression at all
s <- sim(400, 4, 4, frac_de = 0); save_case("null_4v4", s$expr, s$group)
# 6b. strong shifts, both signs
s <- sim(400, 4, 4, frac_de = 0.5, lfc = 2); save_case("shifted_4v4", s$expr, s$group)
# 8. tiny feature counts and degenerate inputs
s <- sim(2, 4, 4); save_case("two_proteins", s$expr, s$group)
s <- sim(3, 4, 4); save_case("three_proteins", s$expr, s$group)
s <- sim(20, 4, 4); e <- s$expr; e[] <- NA; save_case("all_na", e, s$group)
s <- sim(30, 1, 1); save_case("no_residual_df", s$expr, s$group)
cat("done; limma", as.character(packageVersion("limma")), "\n")
