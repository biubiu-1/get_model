library(data.table)
library(ggplot2)
library(dplyr)
library(mgcv)
library(plotly)

setwd('~/SHARE/cellTransformer/evol/')

if (F) {
  df <- fread('./data/250809_HEK293T_hGPK_AAVS1_GFP_single_9-mer_activation.csv')
  colnames(df) <- c("guide_list", "fitness_score", "hits", "local_hits", "fold_change")

  
  ggplot(df, aes(x = local_hits, y = log10(pmax(hits, 1)))) +
    geom_point() +
    geom_smooth(method = "loess") +
    labs(
      title = "Local Hits vs Log10 Global Hits",
      x = "Local Hits",
      y = "Log10 Global Hits"
    ) +
    theme_minimal()
}
 
if (F){ 
  # 1. 准备数据
  df[, log_fold_change := log2(fold_change)]
  df[, log_hits := log10(pmax(hits, 1))]
  
  # 2. 拟合模型
  gam_model <- gam(log_fold_change ~ s(log_hits, local_hits, k = 40), data = df)
  gam.check(gam_model)
  
  # 3. 构建预测网格
  log_hits_seq <- seq(min(df$log_hits), max(df$log_hits), length.out = 100)
  local_hits_seq <- seq(min(df$local_hits), max(df$local_hits), length.out = 100)
  grid <- as.data.table(expand.grid(log_hits = log_hits_seq, local_hits = local_hits_seq))
  
  # 4. 预测
  grid[, gam_fit := predict(gam_model, newdata = grid)]
  
  # 5. 画图
  pdf('./results/250809_HEK293T_hGPK_AAVS1_GFP_single_9-mer_activation.pdf', width = 6, height = 4)
  
  ggplot(grid, aes(x = log_hits, y = local_hits, z = gam_fit)) +
    geom_contour_filled() +
    labs(
      title = "AAVS1-hPGK-GFP1 activation",
      x = "Log10 global hits",
      y = "Local hits",
      fill = "Predicted\nlog2 fold change"
    ) +
    theme_minimal()+
    theme(aspect.ratio = 1)
  
  dev.off()
  
}

if (F) {
  # Reshape grid for 3D surface plot
  z_matrix <- matrix(grid$gam_fit, nrow = 100, ncol = 100, byrow = FALSE)
  
  # Create 3D surface plot
  fig <- plot_ly(
    x = ~log_hits_seq,
    y = ~local_hits_seq,
    z = ~z_matrix,
    type = "surface"
  ) %>%
    layout(
      title = "AAVS1-hPGK-GFP1 Activation",
      scene = list(
        xaxis = list(title = "Log10 global hits"),
        yaxis = list(title = "Local hits"),
        zaxis = list(title = "Log2 fold change"),
        aspectratio = list(x = 1, y = 1, z = 1)
      )
    )
  
  # Save as HTML (interactive plot)
  htmlwidgets::saveWidget(fig, 
     "./results/250809_HEK293T_hGPK_AAVS1_GFP_single_9-mer_activation_3D.html")
}

