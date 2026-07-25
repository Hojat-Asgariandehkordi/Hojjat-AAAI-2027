#!/usr/bin/env Rscript
# ggplot2 bake-off for AAAI results figures (LaTeX / vector PDF+SVG).
# Requires: ggplot2, dplyr, readr, scales, svglite (optional)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
csv_dir <- file.path(root, "_csv")
stopifnot(dir.exists(csv_dir))

need <- c("ggplot2", "dplyr", "readr", "scales")
missing <- need[!vapply(need, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop(
    "Missing R packages: ", paste(missing, collapse = ", "),
    "\nInstall once with:\n  install.packages(c(",
    paste(sprintf('\"%s\"', missing), collapse = ", "),
    "), repos = \"https://cloud.r-project.org\")\n",
    call. = FALSE
  )
}
library(ggplot2)
library(dplyr)
library(readr)
library(scales)

theme_aaai <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.title = element_text(size = base_size),
      plot.title = element_text(face = "bold", size = base_size + 0.5, hjust = 0),
      legend.position = "bottom"
    )
}

cut2 <- function(x) trunc(x * 100) / 100

save_plot <- function(p, out_dir, stem, w = 6.9, h = 2.8) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(out_dir, paste0(stem, ".png")), p, width = w, height = h, dpi = 300)
  if (requireNamespace("svglite", quietly = TRUE)) {
    ggsave(file.path(out_dir, paste0(stem, ".svg")), p, width = w, height = h, device = svglite::svglite)
  }
  message("Wrote ", out_dir)
}

method_cols <- c(
  "SAM-Med2D" = "#4B5563",
  "Ours" = "#0B3D5C",
  "Ours (40-step, R=2)" = "#0B3D5C",
  "Ours (60-step, R=3)" = "#0B3D5C",
  "nnInteractive" = "#6B7280",
  "SegVol" = "#C2410C",
  "VISTA3D" = "#9A3412",
  "SAM-Med3D" = "#78716C"
)

# --- results bars ---
lidc <- read_csv(file.path(csv_dir, "lidc.csv"), show_col_types = FALSE)
fp <- read_csv(file.path(csv_dir, "healthy_fp.csv"), show_col_types = FALSE)
common <- c("SAM-Med2D", "Ours (40-step, R=2)", "nnInteractive", "SAM-Med3D", "SegVol")
dice_map <- setNames(lidc$dice, lidc$method)
# CSV may already use "Ours (40-step, R=2)"; only alias from "Ours" if needed
if (is.na(dice_map[["Ours (40-step, R=2)"]]) && !is.na(dice_map[["Ours"]])) {
  dice_map[["Ours (40-step, R=2)"]] <- dice_map[["Ours"]]
}
fp_map <- setNames(fp$volume_mm3, fp$method)
df_rb <- tibble(
  method = common,
  dice = as.numeric(dice_map[common]),
  volume = as.numeric(fp_map[common])
) %>% mutate(method = factor(method, levels = rev(common)))

p_a <- ggplot(df_rb, aes(dice, method, fill = method)) +
  geom_col(width = 0.72, color = "#111827", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.2f", cut2(dice))), hjust = -0.1, size = 2.6) +
  scale_fill_manual(values = method_cols, guide = "none") +
  scale_x_continuous(limits = c(0, 0.85), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "(a) True nodules (LIDC)", x = "Mean Dice", y = NULL) +
  theme_aaai()
p_b <- ggplot(df_rb, aes(volume, method, fill = method)) +
  geom_col(width = 0.72, color = "#111827", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.0f", volume)), hjust = -0.1, size = 2.6) +
  scale_fill_manual(values = method_cols, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "(b) Healthy FP volume", x = "Pred. volume (mm3)", y = NULL) +
  theme_aaai()

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  p <- p_a + p_b + plot_annotation(title = "Results bars (ggplot2)")
  save_plot(p, file.path(root, "fig_results_bars", "ggplot2"), "fig_results_bars", w = 6.9, h = 2.9)
} else {
  save_plot(p_a, file.path(root, "fig_results_bars", "ggplot2"), "fig_results_bars_a", w = 3.4, h = 2.9)
  save_plot(p_b, file.path(root, "fig_results_bars", "ggplot2"), "fig_results_bars_b", w = 3.4, h = 2.9)
  message("patchwork not installed — wrote panels a/b separately")
}

# --- healthy FP ---
fp2 <- fp %>% mutate(method = factor(method, levels = rev(method)))
p_fp <- ggplot(fp2, aes(volume_mm3, method, fill = as.character(method))) +
  geom_col(width = 0.72, color = "#111827", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.0f", volume_mm3)), hjust = -0.08, size = 2.5) +
  scale_fill_manual(values = method_cols, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Healthy FP probe (ggplot2)", x = "Predicted volume (mm3)", y = NULL) +
  theme_aaai()
save_plot(p_fp, file.path(root, "fig_healthy_fp", "ggplot2"), "fig_healthy_fp", w = 3.5, h = 2.9)

# --- tradeoff scatter ---
tr <- read_csv(file.path(csv_dir, "tradeoff.csv"), show_col_types = FALSE)
if (requireNamespace("ggrepel", quietly = TRUE)) {
  p_tr <- ggplot(tr, aes(fp_volume, lidc_dice, color = method, label = method)) +
    geom_point(size = 3) +
    ggrepel::geom_text_repel(size = 2.4, show.legend = FALSE) +
    scale_color_manual(values = method_cols, guide = "none") +
    labs(title = "FP volume vs LIDC Dice (ggplot2)", x = "Healthy FP volume (mm3)", y = "LIDC Dice") +
    theme_aaai()
} else {
  p_tr <- ggplot(tr, aes(fp_volume, lidc_dice, color = method)) +
    geom_point(size = 3) +
    geom_text(aes(label = method), size = 2.2, vjust = -0.8, show.legend = FALSE) +
    scale_color_manual(values = method_cols, guide = "none") +
    labs(title = "FP volume vs LIDC Dice (ggplot2)", x = "Healthy FP volume (mm3)", y = "LIDC Dice") +
    theme_aaai()
}
save_plot(p_tr, file.path(root, "fig_healthy_fp_tradeoff", "ggplot2"), "fig_healthy_fp_tradeoff", w = 4.2, h = 3.0)

# --- multi-dataset ---
bench <- read_csv(file.path(csv_dir, "bench.csv"), show_col_types = FALSE)
ord <- c("SAM-Med2D", "Ours", "nnInteractive", "SAM-Med3D", "SegVol", "VISTA3D")
bench <- bench %>% mutate(
  method = factor(method, levels = rev(ord)),
  dataset = factor(dataset, levels = c("LIDC", "MAISI", "NSCLC"))
)
p_md <- ggplot(bench, aes(dice, method, fill = dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72, color = "#111827", linewidth = 0.15) +
  scale_fill_manual(values = c(LIDC = "#0B3D5C", MAISI = "#B45309", NSCLC = "#047857")) +
  scale_x_continuous(limits = c(0, 0.9), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Multi-dataset Dice (ggplot2)", x = "Mean Dice", y = NULL, fill = "Dataset") +
  theme_aaai()
save_plot(p_md, file.path(root, "fig_benchmark_dice_3datasets", "ggplot2"), "fig_benchmark_dice_3datasets", w = 6.9, h = 3.1)

# --- ablation ---
abl <- read_csv(file.path(csv_dir, "ablation.csv"), show_col_types = FALSE) %>%
  mutate(config = factor(config, levels = rev(config)))
p_ab <- ggplot(abl, aes(dice, config, fill = dice)) +
  geom_col(width = 0.72, color = "#111827", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.2f", cut2(dice))), hjust = -0.1, size = 2.5) +
  scale_fill_distiller(palette = "Teal", direction = 1, guide = "none") +
  scale_x_continuous(limits = c(0.70, 0.76), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Ablation (ggplot2)", x = "Mean Dice", y = NULL) +
  theme_aaai()
save_plot(p_ab, file.path(root, "fig_ablation_steps", "ggplot2"), "fig_ablation_steps", w = 3.5, h = 2.6)

message("ggplot2 bake-off complete.")
