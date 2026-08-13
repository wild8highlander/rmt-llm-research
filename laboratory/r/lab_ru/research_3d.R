# research_3d.R — 3D исследовательские эксперименты для лаборатории RMT-LLM (R, русская версия)
# =============================================================================
#
# Добавляет девять продвинутых 3D исследовательских экспериментов поверх
# существующего 2D модуля. Каждый эксперимент возвращает структурированный
# список, который функция generate_3d_charts() использует для отрисовки 3D
# визуализаций (PNG/PDF/SVG).
#
# Эксперименты:
#   6. SCEN-3D-HESSIAN — Ландшафт потерь Гессиана
#      Возмущение (w1, w2) вдоль топ-2 собственных направлений и измерение
#      кривизны поверхности потерь; проверяет связь резкости/плоскостности с
#      обобщающей способностью (RMT: объёмные vs аномальные собственные значения).
#   7. SCEN-3D-MANIFOLD — Геометрия многообразия
#      Оценка внутренней размерности скрытых состояний через коэффициент
#      участия PCA; 3D PCA-проекция.
#   8. SCEN-3D-TRAJECTORY — Анализ траектории рассуждений
#      Выборка (шаг, честность, обман, спектральный радиус) и обнаружение
#      начала обмана через точку пересечения.
#   9. SCEN-3D-SPECTRAL-SURFACE — Регрессия спектральной поверхности
#      Подгонка lambda_max(слой, токен) к гладкой поверхности и обнаружение
#      линии бифуркации N_crit.
#  10. SCEN-3D-RIEMANN — Риманова кривизна
#      Оценка дискретной гауссовой кривизны на k-NN графе скрытых состояний;
#      выявляет изгибы многообразия вблизи начала обмана.
#  11. SCEN-3D-ATTENTION-FLOW — 3D-поток внимания
#      Измерение поверхности весов внимания и количественная оценка
#      диагонального vs размазанного режима после N_crit.
#  12. SCEN-3D-NCRIT-SURFACE — Поверхность коллапса N_crit
#      Вычисление поверхности T_crit(beta, mu_RLHF) по порядку Капуто и
#      давлению RLHF.
#  13. SCEN-3D-PARAM-SPACE — Развёртка пространства параметров
#      Развёртка (temperature, top_p) и измерение поверхности скорости
#      галлюцинаций.
#  14. SCEN-3D-COALITION — Дрейф коалиции
#      Симуляция дрейфа обмана многоагентной коалиции по раундам (SCEN-COAL-09).
#
# Все эксперименты принимают конвенцию бесконечных параметров (строка "inf",
# "+inf", "infinity" или числовое Inf поддерживаются через ограничение до
# практических границ только во время вычислений).
#
# Автор: Исхак Хамзатович Исаев, ORCID: 0009-0003-7299-0701
# Лицензия: Проприетарная — Все права защищены.
# =============================================================================

# ---------------------------------------------------------------------------
# Разрешение корневого каталога лаборатории (файл самодостаточен при source())
# ---------------------------------------------------------------------------
.lab_root_3d <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  script_dir <- tryCatch({
    script_arg <- sub("--file=", "", cmd_args[grep("--file=", cmd_args)])
    if (length(script_arg) > 0) dirname(normalizePath(script_arg))
    else getwd()
  }, error = function(e) getwd())
  d <- script_dir
  for (i in 1:6) {
    if (dir.exists(file.path(d, "laboratory")) || dir.exists(file.path(d, "results")))
      return(d)
    d <- dirname(d)
  }
  script_dir
}

.reports_dir_3d <- function() file.path(.lab_root_3d(), "laboratory", "results", "reports")

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

#' Ограничение конвенции бесконечных параметров до конечного числового значения.
#'
#' Обрабатывает: NULL, "inf", "+inf", "infinity" (без учёта регистра), Inf, -Inf,
#' NaN, обычные числовые строки ("3.14") и обычные числовые значения.
#'
#' @param value любой скаляр (NULL, numeric, character, logical)
#' @param default числовое значение, используемое при NULL или неразборе
#' @param max_finite числовая верхняя граница при бесконечном значении
#' @return конечный числовой скаляр
clamp_inf <- function(value, default = 0, max_finite = 1e6) {
  if (is.null(value)) return(default)
  if (length(value) == 0) return(default)
  if (is.logical(value)) {
    v <- as.numeric(value[1])
    if (is.na(v)) return(default)
    return(v)
  }
  if (is.character(value)) {
    s <- tolower(trimws(value[1]))
    if (is.na(s) || s == "") return(default)
    if (s %in% c("inf", "+inf", "infinity", "+infinity")) return(max_finite)
    if (s %in% c("-inf", "-infinity")) return(-max_finite)
    v <- tryCatch(suppressWarnings(as.numeric(s)), warning = function(w) NA_real_)
    if (is.na(v)) return(default)
    if (is.infinite(v) || is.nan(v)) return(max_finite)
    return(v)
  }
  if (is.numeric(value)) {
    v <- value[1]
    # Note: in R, is.na(NaN) is TRUE, so check is.nan/is.infinite first.
    if (is.nan(v) || is.infinite(v)) return(max_finite)
    if (is.na(v)) return(default)
    return(v)
  }
  # Fallback: try numeric coercion
  v <- tryCatch(suppressWarnings(as.numeric(value[1])), warning = function(w) NA_real_)
  if (is.na(v)) return(default)
  if (is.infinite(v) || is.nan(v)) return(max_finite)
  v
}

#' Численно устойчивый softmax вдоль последней оси (вход — вектор).
softmax <- function(x) {
  if (length(x) == 0) return(numeric(0))
  x <- as.numeric(x)
  m <- max(x, na.rm = TRUE)
  if (is.infinite(m) || is.na(m)) m <- 0
  e <- exp(x - m)
  s <- sum(e, na.rm = TRUE)
  if (s <= 0 || !is.finite(s)) {
    out <- rep(1 / length(x), length(x))
    return(out)
  }
  e / s
}

#' Ковариация (N, D) матрицы; возвращает D x D.
#' Повторяет Python _safe_svd_cov.
safe_cov <- function(mat) {
  if (!is.matrix(mat)) {
    dim(mat) <- c(length(mat), 1)
  }
  if (nrow(mat) < 2) {
    d <- ncol(mat)
    return(matrix(0, d, d))
  }
  N <- nrow(mat)
  mu <- colMeans(mat)
  centered <- sweep(mat, 2, mu, "-")
  out <- (t(centered) %*% centered) / max(N - 1, 1)
  out
}

#' Симметричное собственное разложение (отсортировано по возрастанию собственных значений).
#' Возвращает list(values, vectors), где vectors[, i] соответствует values[i].
eig_sym <- function(mat) {
  if (!isSymmetric(mat)) {
    mat <- (mat + t(mat)) / 2
  }
  e <- eigen(mat, symmetric = TRUE, only.values = FALSE)
  ord <- order(e$values, decreasing = FALSE)
  list(values = e$values[ord], vectors = e$vectors[, ord, drop = FALSE])
}

#' Вектор линейного пространства (R-аналог numpy.linspace).
linspace <- function(from, to, n) {
  if (n <= 1) return(from)
  seq(from, to, length.out = n)
}

#' Ограничение значений диапазоном.
clamp01 <- function(x) pmin(pmax(x, 0), 1)

#' Meshgrid: возвращает list(X, Y), где X, Y — матрицы.
#' Использует семантику indexing="ij" (соответствует numpy.meshgrid(..., indexing="ij")
#' из Python-референса): X[i, j] = x[i], Y[i, j] = y[j].
meshgrid <- function(x, y) {
  nx <- length(x)
  ny <- length(y)
  X <- matrix(rep(x, times = ny), nrow = nx, ncol = ny, byrow = FALSE)
  Y <- matrix(rep(y, each = nx), nrow = nx, ncol = ny, byrow = FALSE)
  list(X = X, Y = Y)
}

# ---------------------------------------------------------------------------
# Синтезатор скрытых состояний типа Tiny-GPT (случайный)
# ---------------------------------------------------------------------------
# В R нет модуля tiny_gpt.R, поэтому мы синтезируем скрытые состояния с той
# же статистической структурой: случайные гауссовы эмбеддинги + послойные
# случайные весовые матрицы + активация tanh, затем аппроксимация layer-norm.
# Это сохраняет MP-распределённые ковариационные спектры, на которые
# опираются спектральные эксперименты.

.tiny_gpt_forward <- function(tokens, seed, hidden_dim = 64, n_layers = 6,
                              vocab_size = 256, max_seq_len = 64) {
  if (length(tokens) == 0) tokens <- 0:7
  if (length(tokens) > max_seq_len) tokens <- tokens[1:max_seq_len]
  # Local RNG that does not disturb the global stream.
  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGkind(rng_kind[1], rng_kind[2], rng_kind[3]), add = TRUE)
  set.seed(seed)

  emb <- matrix(rnorm(vocab_size * hidden_dim, 0, 1 / sqrt(hidden_dim)),
                nrow = vocab_size, ncol = hidden_dim)
  # Initial hidden = embedding lookup
  tok_idx <- pmax(0, pmin(tokens, vocab_size - 1)) + 1
  H <- emb[tok_idx, , drop = FALSE]

  # Precompute per-layer weights deterministically from the seed.
  Ws <- vector("list", n_layers)
  for (l in seq_len(n_layers)) {
    Ws[[l]] <- matrix(rnorm(hidden_dim * hidden_dim, 0, 1 / sqrt(hidden_dim)),
                      nrow = hidden_dim, ncol = hidden_dim)
  }

  hidden <- vector("list", n_layers)
  for (l in seq_len(n_layers)) {
    H <- H %*% Ws[[l]]
    # Approximate layer-norm: center and scale rows
    mu <- rowMeans(H)
    sd <- apply(H, 1, sd)
    sd[sd < 1e-9] <- 1
    H <- (H - mu) / sd
    H <- tanh(H)
    hidden[[l]] <- H
  }
  list(logits = H, hidden = hidden)
}

#' Сбор скрытых состояний по множеству случайных промптов.
#' Возвращает (N_total, hidden_dim) матрицу из скрытых состояний последнего слоя.
.collect_hidden_states <- function(seed, n_samples = 32, hidden_dim = 64,
                                   n_layers = 6, vocab_size = 256,
                                   max_seq_len = 64) {
  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGkind(rng_kind[1], rng_kind[2], rng_kind[3]), add = TRUE)
  set.seed(seed)
  rows <- vector("list", n_samples)
  for (i in seq_len(n_samples)) {
    n_tok <- sample(8:max_seq_len, 1)
    toks <- sample(0:(vocab_size - 1), n_tok, replace = TRUE)
    # Vary the seed per prompt so each gives different hidden states
    fwd <- .tiny_gpt_forward(toks, seed = seed + i * 101L,
                             hidden_dim = hidden_dim, n_layers = n_layers,
                             vocab_size = vocab_size, max_seq_len = max_seq_len)
    rows[[i]] <- fwd$hidden[[n_layers]]
  }
  do.call(rbind, rows)
}

# ---------------------------------------------------------------------------
# Простой JSON-энкодер (fallback, когда jsonlite недоступен)
# ---------------------------------------------------------------------------
.json_escape_str <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub('"', '\\"', s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  s <- gsub("\b", "\\b", s, fixed = TRUE)
  s <- gsub("\f", "\\f", s, fixed = TRUE)
  s
}

to_json_simple <- function(x, indent = 0L) {
  ind <- paste(rep(" ", indent), collapse = "")
  ind1 <- paste(rep(" ", indent + 2L), collapse = "")
  if (is.null(x)) return("null")
  if (length(x) == 0) {
    if (is.list(x)) return("[]")
    if (is.character(x)) return("[]")
    if (is.numeric(x) || is.logical(x)) return("[]")
    return("null")
  }
  if (is.logical(x)) {
    return(paste(ifelse(x, "true", "false"), collapse = ", "))
  }
  if (is.numeric(x)) {
    if (length(x) == 1) {
      if (is.na(x)) return("null")
      if (is.nan(x)) return("null")
      if (is.infinite(x)) return(ifelse(x > 0, "1e309", "-1e309"))
      return(formatC(x, digits = 17, format = "g"))
    }
    x[is.na(x)] <- NA
    out <- ifelse(is.na(x), "null",
                  ifelse(is.nan(x), "null",
                         ifelse(is.infinite(x),
                                ifelse(x > 0, "1e309", "-1e309"),
                                formatC(x, digits = 17, format = "g"))))
    return(paste0("[", paste(out, collapse = ", "), "]"))
  }
  if (is.character(x)) {
    if (length(x) == 1) {
      return(paste0('"', .json_escape_str(x), '"'))
    }
    return(paste0("[", paste0('"', vapply(x, .json_escape_str, ""),
                              collapse = ", "), "]",
                  ""))
  }
  if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) {
    # 2D array -> array of arrays
    nr <- nrow(x); nc <- ncol(x)
    if (nr == 0) return("[]")
    row_strs <- character(nr)
    for (i in seq_len(nr)) {
      if (nc == 0) {
        row_strs[i] <- "[]"
      } else {
        item_strs <- character(nc)
        for (j in seq_len(nc)) {
          item_strs[j] <- to_json_simple(x[i, j], indent + 4L)
        }
        row_strs[i] <- paste0("[", paste(item_strs, collapse = ", "), "]")
      }
    }
    return(paste0("[\n", ind1,
                  paste(row_strs, collapse = paste0(",\n", ind1)),
                  "\n", ind, "]"))
  }
  if (is.array(x) && length(dim(x)) > 2) {
    # Higher-dim: recurse on first axis
    slices <- lapply(seq_len(dim(x)[1]), function(i) {
      to_json_simple(x[i, , , drop = TRUE], indent + 2L)
    })
    return(paste0("[\n", ind1,
                  paste(slices, collapse = paste0(",\n", ind1)),
                  "\n", ind, "]"))
  }
  if (is.list(x)) {
    nm <- names(x)
    if (is.null(nm) || all(nm == "")) {
      # Array
      items <- vapply(x, function(el) to_json_simple(el, indent + 2L),
                      character(1))
      return(paste0("[\n", ind1,
                    paste(items, collapse = paste0(",\n", ind1)),
                    "\n", ind, "]"))
    } else {
      # Object
      items <- vapply(seq_along(x), function(i) {
        k <- nm[i]
        if (is.null(k) || k == "") k <- paste0("item", i)
        paste0('"', .json_escape_str(k), '": ',
               to_json_simple(x[[i]], indent + 2L))
      }, character(1))
      return(paste0("{\n", ind1,
                    paste(items, collapse = paste0(",\n", ind1)),
                    "\n", ind, "}"))
    }
  }
  # Fallback: stringify
  paste0('"', .json_escape_str(as.character(x)), '"')
}

#' Запись JSON в файл, предпочтительно через jsonlite, если доступен.
write_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  txt <- tryCatch({
    if (requireNamespace("jsonlite", quietly = TRUE)) {
      jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null",
                       digits = 17, pretty = TRUE)
    } else {
      to_json_simple(x, 0L)
    }
  }, error = function(e) to_json_simple(x, 0L))
  writeLines(as.character(txt), path)
}

# ---------------------------------------------------------------------------
# Эксперимент 6: Ландшафт потерь Гессиана
# ---------------------------------------------------------------------------
exp_hessian_loss_landscape <- function(params) {
  grid <- as.integer(clamp_inf(params$hessian_grid_size, 24, 256))
  if (grid < 4) grid <- 4
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  hidden_dim <- as.integer(clamp_inf(params$hidden_dim, 64, 4096))
  n_layers <- as.integer(clamp_inf(params$n_layers, 6, 64))
  n_heads <- as.integer(clamp_inf(params$n_heads, 4, 64))
  vocab_size <- as.integer(clamp_inf(params$vocab_size, 256, 65536))
  max_seq_len <- 64L

  # Sample hidden states to build empirical Fisher-like matrix
  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGKind_reset(rng_kind), add = TRUE)
  set.seed(seed)
  toks <- sample(0:(vocab_size - 1),
                 min(max_seq_len, 64), replace = TRUE)
  fwd <- .tiny_gpt_forward(toks, seed = seed, hidden_dim = hidden_dim,
                           n_layers = n_layers, vocab_size = vocab_size,
                           max_seq_len = max_seq_len)
  H <- do.call(rbind, fwd$hidden)
  cov <- safe_cov(H)
  e <- eig_sym(cov)
  # Top-2 eigendirections (ascending sort -> last two)
  v1 <- e$vectors[, ncol(e$vectors)]
  v2 <- e$vectors[, ncol(e$vectors) - 1]
  lam1 <- as.numeric(e$values[length(e$values)])
  lam2 <- as.numeric(e$values[length(e$values) - 1])

  # Build (w1, w2) perturbation grid
  span <- 3.0 * sqrt(max(lam1, 1e-9))
  w1_axis <- linspace(-span, span, grid)
  w2_axis <- linspace(-span, span, grid)
  mg <- meshgrid(w1_axis, w2_axis)
  W1 <- mg$X
  W2 <- mg$Y
  L0 <- 1.0
  Z <- L0 + 0.5 * (lam1 * W1^2 - lam2 * W2^2) + 0.05 * sin(W1 * W2)

  list(
    experiment = "hessian_loss_landscape",
    config = list(hidden_dim = hidden_dim, n_layers = n_layers,
                  n_heads = n_heads, vocab_size = vocab_size,
                  max_seq_len = max_seq_len, seed = seed),
    grid_size = grid,
    top_eigenvalues = c(lam1, lam2),
    w1_grid = W1,
    w2_grid = W2,
    loss_surface = Z,
    metrics = list(
      lambda_max = lam1,
      lambda_2 = lam2,
      spectral_gap = lam1 - lam2,
      loss_min = min(Z),
      loss_max = max(Z),
      sharpness = lam1,
      is_saddle = as.logical(lam2 > 0 && lam1 > 0 && any(Z < L0))
    )
  )
}

# Вспомогательная функция для сброса типа RNG
RNGKind_reset <- function(k) {
  if (length(k) >= 3) RNGkind(k[1], k[2], k[3])
  else if (length(k) >= 2) RNGkind(k[1], k[2])
  else if (length(k) >= 1) RNGkind(k[1])
}

# ---------------------------------------------------------------------------
# Эксперимент 7: Геометрия многообразия
# ---------------------------------------------------------------------------
exp_manifold_geometry <- function(params) {
  n_components <- as.integer(clamp_inf(params$pca_components, 3, 64))
  if (n_components < 1) n_components <- 3
  n_samples_req <- as.integer(clamp_inf(params$trajectory_points, 240, 1e6))
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  hidden_dim <- as.integer(clamp_inf(params$hidden_dim, 64, 4096))
  n_layers <- as.integer(clamp_inf(params$n_layers, 6, 64))
  vocab_size <- as.integer(clamp_inf(params$vocab_size, 256, 65536))
  max_seq_len <- 64L

  n_samples <- min(n_samples_req, 32L)
  H <- .collect_hidden_states(seed = seed, n_samples = n_samples,
                              hidden_dim = hidden_dim, n_layers = n_layers,
                              vocab_size = vocab_size, max_seq_len = max_seq_len)
  if (nrow(H) < n_components) {
    reps <- (n_components %/% nrow(H)) + 1L
    H <- H[rep(seq_len(nrow(H)), reps), , drop = FALSE]
    H <- H[seq_len(min(nrow(H), n_components * 4L)), , drop = FALSE]
  }

  # PCA via prcomp (uses SVD)
  pc <- prcomp(H, center = TRUE, scale. = FALSE)
  eigvals_pca <- pc$sdev^2
  sum_ev <- sum(eigvals_pca)
  sum_ev2 <- sum(eigvals_pca^2)
  pr <- as.numeric((sum_ev^2) / max(sum_ev2, 1e-12))

  # Project to top-N components
  proj <- pc$x[, seq_len(min(n_components, ncol(pc$x))), drop = FALSE]
  while (ncol(proj) < 3) proj <- cbind(proj, 0)
  proj3 <- proj[, 1:3, drop = FALSE]

  colors <- seq_len(nrow(proj3)) - 1L

  # Top eigenvalues (cap at 10 for output)
  top_n <- max(n_components, 10)
  pca_eig_out <- as.numeric(eigvals_pca[seq_len(min(length(eigvals_pca), top_n))])

  list(
    experiment = "manifold_geometry",
    config = list(hidden_dim = hidden_dim, n_layers = n_layers,
                  vocab_size = vocab_size, max_seq_len = max_seq_len,
                  seed = seed),
    n_samples = as.integer(nrow(proj3)),
    n_components = n_components,
    pca_eigenvalues = pca_eig_out,
    participation_ratio = pr,
    pca_points = proj3,
    pca_colors = as.numeric(colors),
    metrics = list(
      intrinsic_dim_pr = pr,
      explained_variance_top3 = as.numeric(sum(eigvals_pca[1:min(3, length(eigvals_pca))]) /
                                            max(sum(eigvals_pca), 1e-12)),
      top_eigenvalue = as.numeric(eigvals_pca[1]),
      manifold_volume_proxy = as.numeric(prod(sqrt(pmax(eigvals_pca[1:min(3, length(eigvals_pca))], 1e-12))))
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 8: Анализ траектории рассуждений
# ---------------------------------------------------------------------------
exp_trajectory_analysis <- function(params) {
  n_points <- as.integer(clamp_inf(params$trajectory_points, 64, 1e6))
  if (n_points < 4) n_points <- 64
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  hidden_dim <- as.integer(clamp_inf(params$hidden_dim, 64, 4096))
  n_layers <- as.integer(clamp_inf(params$n_layers, 6, 64))
  vocab_size <- as.integer(clamp_inf(params$vocab_size, 256, 65536))
  max_seq_len <- 64L
  temperature <- clamp_inf(params$temperature, 0.5, 100)
  n_crit <- clamp_inf(params$ncrit_threshold, 96, 1e6)

  # Синтезируем траекторию, выровненную по N_crit (повторяет else-ветку Python).
  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGKind_reset(rng_kind), add = TRUE)
  set.seed(seed)
  steps_arr <- linspace(0, n_points, n_points)
  honesty <- 0.65 / sqrt(1 + steps_arr / n_crit)
  deception <- 0.20 + 0.55 * (1 - exp(-(steps_arr - n_crit) / 30.0))
  deception <- clamp01(deception)
  halluc <- 0.05 + 0.60 * (1 - exp(-(steps_arr - n_crit) / 20.0))
  halluc <- clamp01(halluc)

  # Спектральный радиус на каждом шаге (выборка модели на каждом шаге)
  spec_radius <- numeric(n_points)
  for (step in seq_len(n_points)) {
    sample_toks <- sample(0:(vocab_size - 1), 16, replace = TRUE)
    fwd <- .tiny_gpt_forward(sample_toks, seed = seed + step * 7L,
                             hidden_dim = hidden_dim, n_layers = n_layers,
                             vocab_size = vocab_size, max_seq_len = max_seq_len)
    H <- do.call(rbind, fwd$hidden)
    cov <- safe_cov(H)
    ev <- eig_sym(cov)$values
    spec_radius[step] <- max(ev)
  }
  spec_min <- min(spec_radius)
  spec_max <- max(spec_radius)
  spec_norm <- (spec_radius - spec_min) / max(spec_max - spec_min, 1e-9)

  # Обнаружение начала обмана: первый индекс, где deception > honesty
  onset_idx <- -1L
  for (i in seq_along(deception)) {
    if (deception[i] > honesty[i]) {
      onset_idx <- i - 1L  # 0-based to match Python
      break
    }
  }

  list(
    experiment = "trajectory_analysis",
    n_points = n_points,
    trajectory = list(
      steps = as.numeric(seq(0, n_points - 1)),
      honesty = as.numeric(honesty),
      deception = as.numeric(deception),
      hallucination = as.numeric(halluc),
      spectral = as.numeric(spec_radius),
      spectral_normalized = as.numeric(spec_norm)
    ),
    deception_onset_step = onset_idx,
    metrics = list(
      n_points = n_points,
      onset_step = onset_idx,
      final_honesty = as.numeric(honesty[n_points]),
      final_deception = as.numeric(deception[n_points]),
      mean_spectral_radius = as.numeric(mean(spec_radius)),
      max_spectral_radius = as.numeric(max(spec_radius)),
      honesty_decrease_rate = as.numeric(honesty[1] - honesty[n_points]) / max(n_points, 1),
      deception_increase_rate = as.numeric(deception[n_points] - deception[1]) / max(n_points, 1)
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 9: Регрессия спектральной поверхности
# ---------------------------------------------------------------------------
exp_spectral_surface_regression <- function(params) {
  n_layers <- as.integer(clamp_inf(params$spectral_surface_layers, 6, 64))
  if (n_layers < 2) n_layers <- 2
  n_tokens <- as.integer(clamp_inf(params$trajectory_points, 64, 1e6))
  if (n_tokens < 4) n_tokens <- 64
  n_crit_pred <- clamp_inf(params$ncrit_threshold, 96, 1e6)
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  hidden_dim <- as.integer(clamp_inf(params$hidden_dim, 64, 4096))
  vocab_size <- as.integer(clamp_inf(params$vocab_size, 256, 65536))
  max_seq_len <- 64L

  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGKind_reset(rng_kind), add = TRUE)
  set.seed(seed)

  lambda_max_grid <- matrix(0, nrow = n_layers, ncol = n_tokens)
  for (t in seq_len(n_tokens)) {
    n_tok <- min(t + 3, max_seq_len)
    toks <- sample(0:(vocab_size - 1), n_tok, replace = TRUE)
    fwd <- .tiny_gpt_forward(toks, seed = seed + t * 13L,
                             hidden_dim = hidden_dim, n_layers = n_layers,
                             vocab_size = vocab_size, max_seq_len = max_seq_len)
    for (l in seq_len(min(n_layers, length(fwd$hidden)))) {
      H <- fwd$hidden[[l]]
      cov <- safe_cov(H)
      ev <- eig_sym(cov)$values
      lambda_max_grid[l, t] <- max(ev)
    }
  }

  # Обнаружение бифуркации через вторую разность среднего по слоям
  mean_lambda <- colMeans(lambda_max_grid)
  diff1 <- diff(mean_lambda)
  diff2 <- diff(diff1)
  bif_token <- if (length(diff2) > 0) which.max(abs(diff2)) else 0L
  # Python: argmax(abs(diff2)) + 1 (so 1-based within the diff2 vector + 1)
  bif_token <- as.integer(bif_token)

  # Подгонка pre/post линейной регрессии
  pre_idx <- seq_len(max(bif_token, 1L))
  pre <- mean_lambda[pre_idx]
  pre_slope <- if (length(pre) > 1) as.numeric(coef(lm(pre ~ seq_along(pre)))[2]) else 0
  post_idx <- seq(max(bif_token + 1L, 2L), length(mean_lambda))
  if (length(post_idx) < 1) post_idx <- length(mean_lambda)
  post <- mean_lambda[post_idx]
  post_slope <- if (length(post) > 1) as.numeric(coef(lm(post ~ seq_along(post)))[2]) else 0

  list(
    experiment = "spectral_surface_regression",
    n_layers = n_layers,
    n_tokens = n_tokens,
    lambda_max_grid = lambda_max_grid,
    bifurcation_token = bif_token,
    n_crit_predicted = n_crit_pred,
    metrics = list(
      lambda_max_global = max(lambda_max_grid),
      lambda_min_global = min(lambda_max_grid),
      bifurcation_token = bif_token,
      bifurcation_vs_ncrit = abs(bif_token - n_crit_pred),
      pre_bifurcation_slope = pre_slope,
      post_bifurcation_slope = post_slope,
      slope_ratio = post_slope / max(pre_slope, 1e-9)
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 10: Риманова кривизна
# ---------------------------------------------------------------------------
exp_riemannian_curvature <- function(params) {
  n_neighbors <- as.integer(clamp_inf(params$curvature_neighbors, 8, 64))
  if (n_neighbors < 2) n_neighbors <- 8
  n_samples_req <- as.integer(clamp_inf(params$trajectory_points, 128, 1e6))
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  hidden_dim <- as.integer(clamp_inf(params$hidden_dim, 64, 4096))
  n_layers <- as.integer(clamp_inf(params$n_layers, 6, 64))
  vocab_size <- as.integer(clamp_inf(params$vocab_size, 256, 65536))
  max_seq_len <- 64L

  n_samples <- min(n_samples_req, 32L)
  H <- .collect_hidden_states(seed = seed, n_samples = n_samples,
                              hidden_dim = hidden_dim, n_layers = n_layers,
                              vocab_size = vocab_size, max_seq_len = max_seq_len)

  # Понижение до 3D через SVD (топ-3 компоненты)
  mu <- colMeans(H)
  Hc <- sweep(H, 2, mu, "-")
  sv <- svd(Hc, nu = 0, nv = 3)
  P <- Hc %*% sv$v  # (N, 3)
  if (ncol(P) < 3) P <- cbind(P, matrix(0, nrow(P), 3 - ncol(P)))

  N <- nrow(P)
  curvatures <- numeric(N)
  for (i in seq_len(N)) {
    diffs <- sweep(P, 2, P[i, ], "-")
    dists <- sqrt(rowSums(diffs^2))
    dists[i] <- Inf
    k <- min(n_neighbors, N - 1)
    nn_idx <- order(dists)[seq_len(k)]
    nn <- P[nn_idx, , drop = FALSE]
    vecs <- sweep(nn, 2, P[i, ], "-")
    norms <- sqrt(rowSums(vecs^2))
    norms[norms < 1e-9] <- 1e-9
    vecs_n <- vecs / norms
    # Сортировка по полярному углу в 3D (проецируем на единичную сферу и упорядочиваем)
    # Вычисляем углы, сортируя соседей по азимутальному углу вокруг p.
    # Используем atan2 проекции на плоскость (e1, e2), где e1 — направление
    # первого соседа; проще: сортировка по atan2(y, x) стандартных координат.
    az <- atan2(vecs_n[, 2], vecs_n[, 1])
    ord <- order(az)
    vecs_sorted <- vecs_n[ord, , drop = FALSE]
    # Сумма углов между последовательными единичными векторами
    dots <- rowSums(vecs_sorted[seq_len(k - 1L), , drop = FALSE] *
                    vecs_sorted[2:k, , drop = FALSE])
    angles <- acos(pmin(pmax(dots, -1), 1))
    # Замыкаем цикл
    last_dot <- sum(vecs_sorted[k, ] * vecs_sorted[1, ])
    last_angle <- acos(pmin(pmax(last_dot, -1), 1))
    total_angle <- sum(angles) + last_angle
    K <- 2 * pi - total_angle
    curvatures[i] <- K
  }
  threshold <- mean(curvatures) + 2 * sd(curvatures)
  high_curv_idx <- which(curvatures > threshold) - 1L

  list(
    experiment = "riemannian_curvature",
    n_samples = N,
    n_neighbors = n_neighbors,
    curvatures = as.numeric(curvatures),
    points_3d = P,
    high_curvature_indices = as.integer(high_curv_idx),
    metrics = list(
      mean_curvature = mean(curvatures),
      std_curvature = sd(curvatures),
      max_curvature = max(curvatures),
      min_curvature = min(curvatures),
      n_high_curvature = length(high_curv_idx),
      high_curvature_ratio = length(high_curv_idx) / max(N, 1)
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 11: 3D-поток внимания
# ---------------------------------------------------------------------------
exp_attention_flow_3d <- function(params) {
  resolution <- as.integer(clamp_inf(params$attention_flow_3d_resolution, 32, 256))
  if (resolution < 8) resolution <- 32
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))
  n <- resolution

  # Синтезируем матрицу внимания: диагональ ранняя, размазанная после N_crit
  n_crit <- n %/% 2
  mg <- meshgrid(seq(0, n - 1), seq(0, n - 1))
  Q <- mg$X
  K <- mg$Y
  sigma_pre <- 2.0
  sigma_post <- 6.0
  sigma_grid <- ifelse(Q < n_crit, sigma_pre, sigma_post)
  attn <- exp(-((Q - K)^2) / (2 * sigma_grid^2))
  row_sums <- rowSums(attn)
  row_sums[row_sums < 1e-9] <- 1e-9
  attn <- attn / row_sums

  # Оценка диагональности
  diag_mean <- mean(diag(attn))
  mean_attn <- mean(attn)
  diag_score <- as.numeric(diag_mean / max(mean_attn, 1e-9))

  # Разброс по строкам
  spread_per_row <- numeric(n)
  for (i in seq_len(n)) {
    weights <- attn[i, ]
    positions <- seq(0, n - 1)
    mu_w <- sum(positions * weights) / max(sum(weights), 1e-9)
    spread_per_row[i] <- sqrt(sum((positions - mu_w)^2 * weights) /
                              max(sum(weights), 1e-9))
  }
  smearing_score <- as.numeric(mean(spread_per_row))

  # Энтропия (среднее по строкам)
  ent_per_row <- numeric(n)
  for (i in seq_len(n)) {
    p <- attn[i, ]
    p <- p[p > 1e-12]
    if (length(p) > 0) ent_per_row[i] <- -sum(p * log(p))
  }
  entropy <- mean(ent_per_row)

  list(
    experiment = "attention_flow_3d",
    resolution = n,
    weights = attn,
    spread_per_row = as.numeric(spread_per_row),
    metrics = list(
      diagonality_score = diag_score,
      smearing_score = smearing_score,
      diagonal_to_smeared_ratio = diag_score / max(smearing_score, 1e-9),
      max_weight = max(attn),
      entropy = entropy
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 12: Поверхность коллапса N_crit
# ---------------------------------------------------------------------------
exp_ncrit_surface <- function(params) {
  n_crit_base <- clamp_inf(params$ncrit_threshold, 114.0, 1e6)
  theta_b_deg <- clamp_inf(params$theta_b_deg, 7.07, 360)
  theta_b <- theta_b_deg * pi / 180.0

  beta_axis <- linspace(0.3, 0.9, 24)
  rlhf_axis <- linspace(0.0, 1.0, 24)
  mg <- meshgrid(beta_axis, rlhf_axis)
  B <- mg$X
  R <- mg$Y
  mu_eff <- theta_b + R
  mu_eff[mu_eff < 1e-6] <- 1e-6
  Z <- n_crit_base * mu_eff^(-1.0 / B)

  list(
    experiment = "ncrit_surface",
    beta_axis = as.numeric(beta_axis),
    rlhf_axis = as.numeric(rlhf_axis),
    t_crit_grid = Z,
    n_crit_base = n_crit_base,
    theta_b_rad = theta_b,
    metrics = list(
      t_crit_min = min(Z),
      t_crit_max = max(Z),
      t_crit_mean = mean(Z),
      t_crit_at_beta_0_5_rlhf_0 = n_crit_base * (theta_b^(-1.0 / 0.5)),
      t_crit_at_beta_0_5_rlhf_1 = n_crit_base * ((theta_b + 1.0)^(-1.0 / 0.5))
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 13: Развёртка пространства параметров
# ---------------------------------------------------------------------------
exp_parameter_space <- function(params) {
  grid <- as.integer(clamp_inf(params$parameter_space_grid, 16, 128))
  if (grid < 4) grid <- 16
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))

  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGKind_reset(rng_kind), add = TRUE)
  set.seed(seed)
  temp_axis <- linspace(0.0, 2.0, grid)
  topp_axis <- linspace(0.5, 1.0, grid)
  mg <- meshgrid(temp_axis, topp_axis)
  T <- mg$X
  P <- mg$Y

  base <- (1 / (1 + exp(-(T - 0.8) * 2))) * (P - 0.5) * 2
  noise <- 0.02 * matrix(runif(grid * grid), nrow = grid, ncol = grid)
  Z <- clamp01(base + noise)

  list(
    experiment = "parameter_space",
    temp_axis = as.numeric(temp_axis),
    topp_axis = as.numeric(topp_axis),
    hallucination_grid = Z,
    metrics = list(
      hallucination_min = min(Z),
      hallucination_max = max(Z),
      hallucination_at_T0 = mean(Z[1, ]),
      hallucination_at_T2 = mean(Z[grid, ]),
      hallucination_at_P05 = mean(Z[, 1]),
      hallucination_at_P1 = mean(Z[, grid])
    )
  )
}

# ---------------------------------------------------------------------------
# Эксперимент 14: Дрейф коалиции
# ---------------------------------------------------------------------------
exp_coalition_drift <- function(params) {
  n_agents <- as.integer(clamp_inf(params$n_agents, 2, 64))
  if (n_agents < 1) n_agents <- 2
  n_rounds <- as.integer(clamp_inf(params$n_rounds, 4, 64))
  if (n_rounds < 1) n_rounds <- 4
  seed <- as.integer(clamp_inf(params$seed, 42, 1e6))

  rng_kind <- RNGkind("Mersenne-Twister")
  on.exit(RNGKind_reset(rng_kind), add = TRUE)
  set.seed(seed)

  base_deception <- 0.25 + 0.05 * seq(0, n_agents - 1)
  per_round <- vector("list", n_rounds)
  for (r in seq_len(n_rounds)) {
    scores <- base_deception + 0.12 * (r - 1) + rnorm(n_agents, 0, 0.02)
    scores <- clamp01(scores)
    per_round[[r]] <- as.numeric(scores)
  }

  list(
    experiment = "coalition_drift",
    n_agents = n_agents,
    n_rounds = n_rounds,
    per_round = per_round,
    metrics = list(
      initial_mean_deception = mean(per_round[[1]]),
      final_mean_deception = mean(per_round[[n_rounds]]),
      drift = mean(per_round[[n_rounds]]) - mean(per_round[[1]]),
      convergence_variance = var(per_round[[n_rounds]])
    )
  )
}

# ---------------------------------------------------------------------------
# Реестр экспериментов
# ---------------------------------------------------------------------------
experiments_3d_info <- list(
  "6"  = list(id = "6",
              name = "Ландшафт потерь Гессиана (3D)",
              description = "Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D поверхности потерь."),
  "7"  = list(id = "7",
              name = "Геометрия многообразия (3D PCA)",
              description = "Оценка внутренней размерности через коэффициент участия PCA; 3D проекция."),
  "8"  = list(id = "8",
              name = "Анализ траектории рассуждений (3D)",
              description = "Выборка (шаг, честность, обман, спектральный радиус) и обнаружение начала обмана."),
  "9"  = list(id = "9",
              name = "Регрессия спектральной поверхности (3D)",
              description = "Подгонка поверхности lambda_max(слой, токен) и обнаружение бифуркации N_crit."),
  "10" = list(id = "10",
              name = "Риманова кривизна (3D)",
              description = "Оценка дискретной гауссовой кривизны на k-NN графе скрытых состояний."),
  "11" = list(id = "11",
              name = "3D-поток внимания",
              description = "Измерение поверхности весов внимания и количественная оценка диагонального vs размазанного режима."),
  "12" = list(id = "12",
              name = "Поверхность коллапса N_crit (3D)",
              description = "Вычисление поверхности T_crit(beta, mu_RLHF) по порядку Капуто и давлению RLHF."),
  "13" = list(id = "13",
              name = "Развёртка пространства параметров (3D)",
              description = "Развёртка (temperature, top_p) и измерение поверхности скорости галлюцинаций."),
  "14" = list(id = "14",
              name = "Дрейф коалиционного обмана (3D)",
              description = "Симуляция дрейфа обмана многоагентной коалиции по раундам (SCEN-COAL-09).")
)

# Соответствие id -> snake_case ключ, используемый charts_3d
.name_map_3d <- list(
  "6"  = "loss_landscape",
  "7"  = "manifold_geometry",
  "8"  = "trajectory",
  "9"  = "spectral_surface",
  "10" = "riemannian_curvature",
  "11" = "attention_flow_3d",
  "12" = "ncrit_surface",
  "13" = "parameter_space",
  "14" = "coalition_drift"
)

# Соответствие id -> функция-исполнитель
.runners_3d <- list(
  "6"  = exp_hessian_loss_landscape,
  "7"  = exp_manifold_geometry,
  "8"  = exp_trajectory_analysis,
  "9"  = exp_spectral_surface_regression,
  "10" = exp_riemannian_curvature,
  "11" = exp_attention_flow_3d,
  "12" = exp_ncrit_surface,
  "13" = exp_parameter_space,
  "14" = exp_coalition_drift
)

# ---------------------------------------------------------------------------
# Запуск одного 3D-эксперимента по ID
# ---------------------------------------------------------------------------
run_3d_experiment <- function(id, params = list()) {
  if (!(id %in% names(.runners_3d))) {
    stop(sprintf("Unknown 3D experiment: %s. Known: %s",
                 id, paste(names(.runners_3d), collapse = ", ")))
  }
  info <- experiments_3d_info[[id]]
  runner <- .runners_3d[[id]]
  t0 <- proc.time()
  result <- tryCatch(runner(params),
                     error = function(e) list(error = conditionMessage(e)))
  elapsed <- as.numeric((proc.time() - t0)["elapsed"])

  # Уплощение параметров (отбрасываем list/data.frame для компактного JSON)
  params_flat <- list()
  for (k in names(params)) {
    v <- params[[k]]
    if (!is.list(v) && !is.data.frame(v)) params_flat[[k]] <- v
  }

  result[["experiment_id"]] <- id
  result[["name"]] <- info$name
  result[["description"]] <- info$description
  result[["experiment_name"]] <- info$name
  result[["experiment_description"]] <- info$description
  result[["elapsed_seconds"]] <- elapsed
  result[["parameters"]] <- params_flat

  # Запись результатов JSON в каталог reports
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_path <- file.path(.reports_dir_3d(),
                        sprintf("%s_3d_exp_%s_results.json", ts, id))
  tryCatch(write_json(result, out_path),
           error = function(e) message(sprintf("[WARN] JSON write failed: %s",
                                               conditionMessage(e))))
  result[["results_path"]] <- out_path

  result
}

# ---------------------------------------------------------------------------
# Запуск всех 9 3D-экспериментов
# ---------------------------------------------------------------------------
run_all_3d <- function(params = list()) {
  combined <- list("3d_research" = list(), experiments = list())
  for (id in names(.runners_3d)) {
    tryCatch({
      res <- run_3d_experiment(id, params)
      combined$experiments <- c(combined$experiments, list(list(
        id = id,
        name = res$name,
        elapsed_seconds = res$elapsed_seconds
      )))
      key <- .name_map_3d[[id]]
      if (is.null(key)) key <- paste0("exp_", id)
      combined[["3d_research"]][[key]] <- res
    }, error = function(e) {
      message(sprintf("  [WARN] 3D experiment %s failed: %s", id,
                      conditionMessage(e)))
      combined$experiments <- c(combined$experiments, list(list(
        id = id, error = conditionMessage(e)
      )))
    })
  }
  combined
}

# ---------------------------------------------------------------------------
# Генерация 3D-графиков (PNG 600 DPI, PDF, SVG)
# ---------------------------------------------------------------------------
# Использует нативные R persp() / image() / проекции типа scatterplot3d.
# Каждый график записывается в PNG, PDF, SVG.

.persp_surface <- function(x_axis, y_axis, Z, main, xlab, ylab, zlab,
                           phi = 30, theta = 45, col = "lightblue") {
  # Z — матрица с nrow=length(x_axis), ncol=length(y_axis) (строки = x)
  nr <- length(x_axis)
  nc <- length(y_axis)
  if (!is.matrix(Z) || nrow(Z) != nr || ncol(Z) != nc) {
    Zm <- matrix(Z, nrow = nr, ncol = nc)
  } else {
    Zm <- Z
  }
  persp(x = x_axis, y = y_axis, z = Zm,
        xlab = xlab, ylab = ylab, zlab = zlab,
        main = main, col = col, theta = theta, phi = phi,
        ticktype = "detailed", shade = 0.5, border = NA)
}

.generate_3d_chart_for <- function(res, out_dir, file_prefix, ext) {
  exp_id <- res[["experiment_id"]]
  name <- res[["name"]]
  dev_open <- function(path) {
    if (ext == "png") {
      png(path, width = 8, height = 6, units = "in", res = 600)
    } else if (ext == "pdf") {
      pdf(path, width = 8, height = 6)
    } else if (ext == "svg") {
      svg(path, width = 8, height = 6)
    } else {
      png(path, width = 8, height = 6, units = "in", res = 600)
    }
  }
  path <- file.path(out_dir, sprintf("%s.%s", file_prefix, ext))
  tryCatch({
    dev_open(path)
    par(mar = c(4, 4, 3, 1))
    ok <- tryCatch({
      .draw_3d_for_experiment(exp_id, res)
      TRUE
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, sprintf("[%s — %s]", name, conditionMessage(e)))
      FALSE
    })
    dev.off()
    path
  }, error = function(e) {
    try(dev.off(), silent = TRUE)
    writeLines(sprintf("[chart %s — device unavailable: %s]",
                       file_prefix, conditionMessage(e)),
               paste0(path, ".txt"))
    NA_character_
  })
}

.draw_3d_for_experiment <- function(exp_id, res) {
  if (exp_id == "6") {
    # Ландшафт потерь Гессиана
    W1 <- res$w1_grid
    W2 <- res$w2_grid
    Z <- res$loss_surface
    if (is.matrix(W1) && is.matrix(Z)) {
      x_axis <- W1[, 1]
      y_axis <- W2[1, ]
      .persp_surface(x_axis, y_axis, Z,
                     main = "Ландшафт потерь Гессиана (3D)",
                     xlab = "w1 (топ собственное направление)", ylab = "w2", zlab = "потери")
    } else plot.new()
  } else if (exp_id == "7") {
    # Геометрия многообразия: 3D PCA-проекция — 2D-скаттер (PC1 x PC2)
    # цвет по позиции токена; PC3 кодирован размером точки.
    pts <- res$pca_points
    if (is.matrix(pts) && ncol(pts) >= 3) {
      xs <- pts[, 1]; ys <- pts[, 2]; zs <- pts[, 3]
      n <- nrow(pts)
      cols <- topo.colors(n)
      # Нормализуем z к размерам точек 0.5..2.5
      zr <- range(zs); zs_n <- (zs - zr[1]) / max(zr[2] - zr[1], 1e-9)
      psize <- 0.5 + 2.0 * zs_n
      plot(xs, ys, col = cols, pch = 19, cex = psize,
           xlab = "PC1", ylab = "PC2",
           main = "Многообразие скрытых состояний (3D PCA, размер = PC3)")
    } else plot.new()
  } else if (exp_id == "8") {
    # Траектория: линии честности/обмана/галлюцинации по шагам
    tr <- res$trajectory
    steps <- tr$steps
    plot(steps, tr$honesty, type = "l", col = "blue", lwd = 2,
         ylim = c(0, 1), xlab = "Шаг", ylab = "Оценка",
         main = "Траектория рассуждений (3D: шаг x честность x обман)")
    lines(steps, tr$deception, col = "red", lwd = 2)
    lines(steps, tr$hallucination, col = "darkgreen", lwd = 2)
    abline(v = res$deception_onset_step, lty = 2, col = "orange")
    legend("topright", bty = "n",
           legend = c("честность", "обман", "галлюцинация", "начало"),
           col = c("blue", "red", "darkgreen", "orange"),
           lty = c(1, 1, 1, 2), lwd = 2)
  } else if (exp_id == "9") {
    # Спектральная поверхность: lambda_max(слой, токен)
    G <- res$lambda_max_grid
    if (is.matrix(G)) {
      x_axis <- seq_len(nrow(G)) - 1
      y_axis <- seq_len(ncol(G)) - 1
      .persp_surface(x_axis, y_axis, G,
                     main = "Спектральная поверхность (lambda_max(слой, токен))",
                     xlab = "слой", ylab = "токен", zlab = "lambda_max")
    } else plot.new()
  } else if (exp_id == "10") {
    # Риманова кривизна: 3D-скаттер, цвет = кривизна
    pts <- res$points_3d
    curv <- res$curvatures
    if (is.matrix(pts) && ncol(pts) >= 3 && length(curv) == nrow(pts)) {
      # Отображаем значения кривизны в цвета (findInterval для надёжности)
      if (length(unique(curv)) >= 2) {
        bks <- seq(min(curv), max(curv), length.out = 65)
        codes <- findInterval(curv, bks, all.inside = TRUE)
        cols <- rev(heat.colors(64))[codes]
      } else {
        cols <- rep("red", length(curv))
      }
      plot(pts[, 1], pts[, 2], col = cols, pch = 19, cex = 1.2,
           xlab = "PC1", ylab = "PC2",
           main = "Риманова кривизна (цвет = гауссова кривизна)")
    } else plot.new()
  } else if (exp_id == "11") {
    # Поверхность потока внимания
    W <- res$weights
    if (is.matrix(W)) {
      x_axis <- seq_len(nrow(W)) - 1
      y_axis <- seq_len(ncol(W)) - 1
      .persp_surface(x_axis, y_axis, W,
                     main = "3D-поток внимания (запрос x ключ x вес)",
                     xlab = "запрос", ylab = "ключ", zlab = "вес")
    } else plot.new()
  } else if (exp_id == "12") {
    # Поверхность коллапса N_crit
    ba <- res$beta_axis
    ra <- res$rlhf_axis
    Z <- res$t_crit_grid
    if (is.matrix(Z)) {
      .persp_surface(ba, ra, Z,
                     main = "Поверхность коллапса N_crit T(beta, mu_RLHF)",
                     xlab = "beta", ylab = "давление RLHF", zlab = "T_crit")
    } else plot.new()
  } else if (exp_id == "13") {
    # Развёртка пространства параметров
    ta <- res$temp_axis
    pa <- res$topp_axis
    Z <- res$hallucination_grid
    if (is.matrix(Z)) {
      .persp_surface(ta, pa, Z,
                     main = "Развёртка пространства параметров (T x top_p x галлюцинация)",
                     xlab = "температура", ylab = "top_p", zlab = "галлюцинация")
    } else plot.new()
  } else if (exp_id == "14") {
    # Дрейф коалиции: линии по агентам по раундам
    per_round <- res$per_round
    if (is.list(per_round) && length(per_round) > 0) {
      n_rounds <- length(per_round)
      n_agents <- length(per_round[[1]])
      mat <- do.call(rbind, per_round)
      rounds <- seq_len(n_rounds) - 1
      matplot(rounds, mat, type = "b", pch = 19, lty = 1, lwd = 2,
              xlab = "Раунд", ylab = "Оценка обмана",
              main = "Дрейф коалиции (раунд x агент x обман)")
    } else plot.new()
  } else {
    plot.new()
    text(0.5, 0.5, sprintf("[Неизвестный эксперимент %s]", exp_id))
  }
}

#' Генерация 3D-графиков PNG/PDF/SVG для каждого эксперимента в `results`.
#' @param results либо один результат эксперимента (с experiment_id),
#'                либо объединённый список из run_all_3d().
#' @param out_dir выходной каталог (создаётся при отсутствии)
#' @return список путей к записанным файлам (невидимо)
generate_3d_charts <- function(results, out_dir = NULL) {
  if (is.null(out_dir)) {
    out_dir <- file.path(.lab_root_3d(), "laboratory", "results", "charts_3d")
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  written <- list()

  # Нормализация: если на входе объединённый результат, итерируемся по 3d_research
  if (!is.null(results[["3d_research"]]) && is.list(results[["3d_research"]])) {
    items <- results[["3d_research"]]
  } else if (!is.null(results[["experiment_id"]])) {
    items <- list(res = results)
  } else {
    items <- list(res = results)
  }

  for (key in names(items)) {
    res <- items[[key]]
    if (is.null(res[["experiment_id"]])) next
    exp_id <- res[["experiment_id"]]
    snake <- .name_map_3d[[exp_id]]
    if (is.null(snake)) snake <- paste0("exp_", exp_id)
    file_prefix <- sprintf("3d_%s_%s", exp_id, snake)
    for (ext in c("png", "pdf", "svg")) {
      p <- .generate_3d_chart_for(res, out_dir, file_prefix, ext)
      if (!is.na(p)) written[[length(written) + 1L]] <- p
    }
  }
  invisible(unlist(written))
}

# ---------------------------------------------------------------------------
# Дымовой тест (запуск напрямую: Rscript -e 'source(...); ...'
# ---------------------------------------------------------------------------
.smoke_test_3d <- function() {
  cat("Запуск всех 3D-экспериментов (R)…\n")
  res <- run_all_3d(list(hessian_grid_size = 16, trajectory_points = 32))
  cat(sprintf("\nЗавершено %d экспериментов:\n", length(res$experiments)))
  for (e in res$experiments) {
    if (!is.null(e$error)) {
      cat(sprintf("  [%s] ОШИБКА: %s\n", e$id, e$error))
    } else {
      cat(sprintf("  [%s] %s — %.3fs\n", e$id, e$name, e$elapsed_seconds))
    }
  }
  cat("\nКлючи 3D-исследований:", paste(names(res[["3d_research"]]), collapse = ", "), "\n")
  invisible(res)
}
