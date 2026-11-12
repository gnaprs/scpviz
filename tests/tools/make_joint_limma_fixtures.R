# Reference outputs from R limma for validating scpviz joint_de (multi-group, contrasts).
# Dev/test-only (R is NOT a runtime dependency). Re-run only to regenerate fixtures:
#   Rscript tests/tools/make_joint_limma_fixtures.R
# Model: y ~ 0 + group; contrasts.fit(); eBayes(trend = FALSE, robust = FALSE).
suppressMessages(library(limma))
out_dir <- file.path("tests", "data", "joint_limma")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20261112)

sim <- function(sizes, n_prot = 300, s0sq = 0.1, d0 = 4, shifts = NULL, na_frac = 0) {
  group <- rep(names(sizes), sizes)
  s2 <- s0sq * d0 / rchisq(n_prot, df = d0)
  e <- matrix(rnorm(n_prot * length(group)), n_prot) * sqrt(s2) + rnorm(n_prot, 20, 2)
  for (g in names(shifts)) e[seq_len(30), group == g] <- e[seq_len(30), group == g] + shifts[[g]]
  if (na_frac > 0) e[matrix(runif(length(e)) < na_frac, n_prot)] <- NA
  dimnames(e) <- list(sprintf("P%04d", seq_len(n_prot)), sprintf("s%d", seq_along(group)))
  list(expr = e, group = group)
}

contrast_matrix <- function(levels, specs) {  # specs: named list of named numeric weight vectors
  C <- matrix(0, length(levels), length(specs), dimnames = list(levels, names(specs)))
  for (nm in names(specs)) C[names(specs[[nm]]), nm] <- specs[[nm]]
  C
}

save_case <- function(name, s, specs) {
  levels <- sort(unique(s$group))
  C <- contrast_matrix(levels, specs)
  g <- factor(s$group, levels = levels)
  design <- model.matrix(~ 0 + g); colnames(design) <- levels
  fit <- eBayes(contrasts.fit(lmFit(s$expr, design), C), trend = FALSE, robust = FALSE)
  write.csv(s$expr, file.path(out_dir, paste0(name, "_expr.csv")), na = "NA")
  write.csv(data.frame(sample = colnames(s$expr), group = s$group), file.path(out_dir, paste0(name, "_group.csv")), row.names = FALSE)
  write.csv(as.data.frame(C), file.path(out_dir, paste0(name, "_contrasts.csv")))
  write.csv(data.frame(s2_prior = fit$s2.prior[1], df_prior = fit$df.prior[1], df_pooled = sum(fit$df.residual, na.rm = TRUE),
                       limma_version = as.character(packageVersion("limma")), r_version = R.version.string),
            file.path(out_dir, paste0(name, "_prior.csv")), row.names = FALSE)
  for (j in seq_len(ncol(C))) {
    res <- data.frame(protein = rownames(s$expr), coef = fit$coefficients[, j], stdev_unscaled = fit$stdev.unscaled[, j],
                      s2 = fit$sigma^2, df_residual = fit$df.residual, s2_post = fit$s2.post,
                      t = fit$t[, j], df_total = fit$df.total, p_value = fit$p.value[, j])
    res$p_adj <- p.adjust(res$p_value, "BH")
    write.csv(res, file.path(out_dir, sprintf("%s_c%d_result.csv", name, j)), row.names = FALSE, na = "NA")
  }
}

pair <- function(a, b) setNames(c(1, -1), c(a, b))

# 1. balanced 3 groups x 4, complete
save_case("balanced3", sim(c(a = 4, b = 4, c = 4), shifts = list(b = 1.5)),
          list(b_vs_a = pair("b", "a"), c_vs_a = pair("c", "a")))
# 2. PISA-like six unbalanced groups (n = 16, df = 10)
save_case("pisa6", sim(c(a = 3, b = 2, c = 3, d = 3, e = 3, f = 2), shifts = list(b = 2)),
          list(b_vs_a = pair("b", "a"), d_vs_c = pair("d", "c"), f_vs_e = pair("f", "e")))
# 3. vehicle + 3 doses, plus an interaction-style weight vector
save_case("doses4", sim(c(veh = 4, d1 = 4, d2 = 4, d3 = 4), shifts = list(d3 = 2)),
          list(d1 = pair("d1", "veh"), d2 = pair("d2", "veh"), d3 = pair("d3", "veh"),
               inter = c(d1 = 1, veh = -1, d2 = -1, d3 = 1)))
# 4. missing values: unequal df, a group absent for some proteins, proteins with df 0
s <- sim(c(a = 4, b = 4, c = 4, d = 3), na_frac = 0.12, shifts = list(c = 1.5))
s$expr[1:5, s$group == "a"] <- NA                  # group a absent for P1-P5
s$expr[6:8, s$group != "c"] <- NA                  # only group c observed
s$expr[9, ] <- NA                                  # entirely missing
s$expr[10, ] <- c(rep(NA, 10), 1, NA, NA, 2)[seq_len(ncol(s$expr))]  # two observations, two groups (df 0)
save_case("missing4", s,
          list(b_vs_a = pair("b", "a"), c_vs_b = pair("c", "b"), inter = c(a = 1, b = -1, c = -1, d = 1)))
