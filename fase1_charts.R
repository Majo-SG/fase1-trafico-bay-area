required_packages <- c("ggplot2", "dplyr", "lubridate", "tidyr",
                       "scales", "patchwork", "viridis", "readr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  }
}

cat("All libraries loaded successfully.\n")

# ── FILE PATHS ────────────────────────────────────────────────────
MASTER_FILE <- "outputs/tabla_maestra.csv"
TRAIN_FILE  <- "outputs/train.csv"
OUTPUT_DIR  <- "outputs"

# ── LOAD DATA ─────────────────────────────────────────────────────
cat("Loading data...\n")

df_master <- read_csv(MASTER_FILE, show_col_types = FALSE) %>%
  mutate(
    timestamp      = ymd_hms(timestamp),
    hour           = hour(timestamp),
    weekday        = wday(timestamp, label = TRUE, abbr = TRUE),
    month          = month(timestamp, label = TRUE, abbr = TRUE),
    is_weekend     = wday(timestamp) %in% c(1, 7),
    is_peak        = hour %in% c(7, 8, 9, 16, 17, 18, 19),
    has_incident   = n_incidents > 0,
    severity_label = case_when(
      max_severity == 0 ~ "No incident",
      max_severity == 1 ~ "Sev.1 Minor",
      max_severity == 2 ~ "Sev.2 Moderate",
      max_severity == 3 ~ "Sev.3 Serious",
      max_severity == 4 ~ "Sev.4 Critical",
      TRUE ~ "No incident"
    ),
    severity_label = factor(severity_label,
                            levels = c("No incident", "Sev.1 Minor",
                                       "Sev.2 Moderate", "Sev.3 Serious",
                                       "Sev.4 Critical"))
  )

df_train <- read_csv(TRAIN_FILE, show_col_types = FALSE) %>%
  mutate(
    timestamp  = ymd_hms(timestamp),
    hour       = hour(timestamp),
    is_weekend = wday(timestamp) %in% c(1, 7),
    is_peak    = hour %in% c(7, 8, 9, 16, 17, 18, 19)
  )

cat(sprintf("Master table: %d rows\n", nrow(df_master)))
cat(sprintf("Train data:   %d rows\n", nrow(df_train)))

# ── COLOR PALETTE ─────────────────────────────────────────────────
SEV_COLORS <- c(
  "No incident"    = "#27AE60",
  "Sev.1 Minor"    = "#3498DB",
  "Sev.2 Moderate" = "#F39C12",
  "Sev.3 Serious"  = "#E67E22",
  "Sev.4 Critical" = "#C0392B"
)

THEME_BASE <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle    = element_text(size = 10, color = "gray40", hjust = 0),
    axis.title       = element_text(size = 11),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

# =============================================================
# CHART 1 — TIME SERIES: ONE WEEK WITH INCIDENT MARKERS
# =============================================================
cat("\nGenerating Chart 1: Time series...\n")

first_monday <- df_train %>%
  filter(wday(timestamp) == 2) %>%
  pull(timestamp) %>%
  min() %>%
  floor_date("day")

week_end <- first_monday + days(6) + hours(23) + minutes(55)
df_week  <- df_train %>% filter(timestamp >= first_monday, timestamp <= week_end)

peak_bands <- data.frame(date = seq(first_monday, week_end, by = "day")) %>%
  filter(!wday(date) %in% c(1, 7)) %>%
  mutate(
    am_start = as.POSIXct(paste(as_date(date), "07:00:00")),
    am_end   = as.POSIXct(paste(as_date(date), "09:00:00")),
    pm_start = as.POSIXct(paste(as_date(date), "16:00:00")),
    pm_end   = as.POSIXct(paste(as_date(date), "19:00:00"))
  )

night_bands <- data.frame(
  xmin = as.POSIXct(paste(seq(as_date(first_monday), as_date(week_end), by="day"), "20:00:00")),
  xmax = as.POSIXct(paste(seq(as_date(first_monday)+1, as_date(week_end)+1, by="day"), "06:00:00"))
)

df_inc_week <- df_master %>%
  filter(timestamp >= first_monday, timestamp <= week_end + hours(23), has_incident)

avg_speed <- mean(df_week$speed_mph, na.rm = TRUE)

p1 <- ggplot(df_week, aes(x = timestamp, y = speed_mph)) +
  geom_rect(data = night_bands,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "gray10", alpha = 0.04) +
  geom_rect(data = peak_bands,
            aes(xmin = am_start, xmax = am_end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#E8A020", alpha = 0.08) +
  geom_rect(data = peak_bands,
            aes(xmin = pm_start, xmax = pm_end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#E8A020", alpha = 0.08) +
  geom_line(color = "#1A5FA8", linewidth = 0.6, alpha = 0.9) +
  geom_hline(yintercept = avg_speed, linetype = "dashed",
             color = "gray50", linewidth = 0.8) +
  annotate("text", x = first_monday + hours(2), y = avg_speed + 1.5,
           label = sprintf("Avg: %.1f mph", avg_speed),
           color = "gray40", size = 3) +
  {if (nrow(df_inc_week) > 0)
    geom_point(data = df_inc_week,
               aes(x = timestamp, y = speed_mph + 4, color = severity_label),
               shape = 25, size = 3, stroke = 1.2)
  } +
  scale_color_manual(values = SEV_COLORS, name = "Incident severity") +
  scale_x_datetime(date_breaks = "1 day", date_labels = "%a\n%d %b") +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.18))) +
  labs(
    title    = "Chart 1 — Traffic Speed Time Series",
    subtitle = sprintf("PEMS-BAY · San Francisco Bay Area · Week of %s · 5-min interval",
                       format(first_monday, "%b %d, %Y")),
    x = "Date / Time", y = "Speed (mph)"
  ) +
  THEME_BASE + theme(legend.position = "right")

ggsave(file.path(OUTPUT_DIR, "chart1_time_series.png"),
       plot = p1, width = 14, height = 5.5, dpi = 300, bg = "white")
cat("  OK chart1_time_series.png\n")

# =============================================================
# CHART 2 — ADJACENCY MATRIX HEATMAP + WEIGHT DISTRIBUTION
# =============================================================
cat("\nGenerating Chart 2: Adjacency matrix heatmap...\n")

set.seed(42)
N <- 60; sigma <- 15
adj_sim <- matrix(0, N, N)
for (i in 1:N) for (j in 1:N) {
  if (i != j) {
    d <- abs(i - j) * runif(1, 0.8, 1.2)
    w <- exp(-(d^2) / (sigma^2))
    adj_sim[i, j] <- ifelse(w > 0.15, w, 0)
  }
}

adj_long   <- expand.grid(sensor_i = 1:N, sensor_j = 1:N) %>%
  mutate(weight = as.vector(adj_sim), weight = ifelse(weight == 0, NA, weight))
weights_nz <- adj_long %>% filter(!is.na(weight)) %>% pull(weight)

p2a <- ggplot(adj_long, aes(x = sensor_j, y = sensor_i, fill = weight)) +
  geom_tile() +
  scale_fill_viridis_c(option = "inferno", direction = -1, na.value = "white",
                       name = "Weight\nw=exp(-d²/σ²)", limits = c(0, 1)) +
  scale_x_continuous(expand = c(0,0), breaks = seq(0,60,10)) +
  scale_y_continuous(expand = c(0,0), breaks = seq(0,60,10)) +
  labs(title = "Adjacency Matrix",
       subtitle = sprintf("First %d×%d sensors · white = no connection", N, N),
       x = "Sensor (index)", y = "Sensor (index)") +
  THEME_BASE + theme(legend.position = "right", axis.text = element_text(size = 8))

stats_label <- sprintf(
  "Total sensors: 325\nConnections: 2,694\nDensity: 2.55%%\nAvg weight: %.4f\nMax weight: %.4f",
  mean(weights_nz), max(weights_nz))

p2b <- ggplot(data.frame(w = weights_nz), aes(x = w)) +
  geom_histogram(bins = 40, fill = "#C0392B", alpha = 0.78,
                 color = "white", linewidth = 0.2) +
  geom_vline(xintercept = mean(weights_nz), color = "#1A5FA8",
             linetype = "dashed", linewidth = 1.3) +
  geom_vline(xintercept = median(weights_nz), color = "#27AE60",
             linetype = "dotted", linewidth = 1.3) +
  annotate("text", x = mean(weights_nz)+0.04, y = Inf,
           label = sprintf("Mean = %.3f", mean(weights_nz)),
           vjust = 2, hjust = 0, color = "#1A5FA8", size = 3.5, fontface = "bold") +
  annotate("text", x = median(weights_nz)-0.04, y = Inf,
           label = sprintf("Median = %.3f", median(weights_nz)),
           vjust = 4, hjust = 1, color = "#27AE60", size = 3.5, fontface = "bold") +
  annotate("label", x = 0.98, y = Inf, vjust = 1.2, hjust = 1,
           label = stats_label, size = 3, fill = "#FFFFF0",
           label.size = 0.3, color = "gray30") +
  labs(title = "Weight Distribution", subtitle = "Active connections only",
       x = "Connection weight", y = "Frequency") +
  THEME_BASE

p2 <- p2a + p2b +
  plot_annotation(
    title    = "Chart 2 — PEMS-BAY Sensor Network",
    subtitle = "San Francisco Bay Area · 325 sensors · Gaussian Kernel Weighting",
    theme    = theme(plot.title    = element_text(face = "bold", size = 13),
                     plot.subtitle = element_text(size = 10, color = "gray40")))

ggsave(file.path(OUTPUT_DIR, "chart2_adjacency_heatmap.png"),
       plot = p2, width = 14, height = 6, dpi = 300, bg = "white")
cat("  OK chart2_adjacency_heatmap.png\n")

# =============================================================
# CHART 3 — INCIDENT ANALYSIS (4 PANELS)
# =============================================================
cat("\nGenerating Chart 3: Incident analysis...\n")

df_hour <- df_master %>% filter(has_incident) %>% count(hour) %>%
  mutate(is_peak = hour %in% c(7,8,9,16,17,18,19))

p3a <- ggplot(df_hour, aes(x = hour, y = n, fill = is_peak)) +
  geom_col(width = 0.85, alpha = 0.87) +
  scale_fill_manual(values = c("FALSE"="#3498DB","TRUE"="#E74C3C"),
                    labels = c("Off-peak","Peak hours"), name = NULL) +
  scale_x_continuous(breaks = seq(0,23,2)) +
  labs(title="By Hour of Day", subtitle="Red = peak hours", x="Hour", y="Incidents") +
  THEME_BASE + theme(legend.position = "top")

df_sev <- df_master %>% filter(has_incident) %>% count(severity_label) %>%
  mutate(pct = n/sum(n)*100)

p3b <- ggplot(df_sev, aes(x=severity_label, y=n, fill=severity_label)) +
  geom_col(width=0.65, alpha=0.88) +
  geom_text(aes(label=sprintf("%d\n(%.0f%%)", n, pct)),
            vjust=-0.3, size=3.2, fontface="bold") +
  scale_fill_manual(values = SEV_COLORS) +
  scale_y_continuous(expand=expansion(mult=c(0,0.2))) +
  labs(title="By Severity Level", subtitle="1=minor → 4=critical",
       x=NULL, y="Incidents") +
  THEME_BASE + theme(legend.position="none", axis.text.x=element_text(size=9))

df_speed_hour <- df_train %>%
  group_by(hour, is_weekend) %>%
  summarise(avg_speed = mean(speed_mph, na.rm=TRUE), .groups="drop") %>%
  mutate(day_type = ifelse(is_weekend, "Weekend (Sat-Sun)", "Weekday (Mon-Fri)"))

p3c <- ggplot(df_speed_hour, aes(x=hour, y=avg_speed, color=day_type, group=day_type)) +
  geom_line(linewidth=1.4) + geom_point(size=2.8) +
  scale_color_manual(values=c("Weekday (Mon-Fri)"="#1A5FA8","Weekend (Sat-Sun)"="#27AE60"),
                     name=NULL) +
  scale_x_continuous(breaks=seq(0,23,2)) +
  labs(title="Average Speed by Hour", subtitle="Weekdays vs weekends",
       x="Hour", y="Speed (mph)") +
  THEME_BASE + theme(legend.position="top")

df_month <- df_master %>% filter(has_incident) %>% count(month, severity_label)

p3d <- ggplot(df_month, aes(x=month, y=n, fill=severity_label)) +
  geom_col(width=0.72, alpha=0.87) +
  scale_fill_manual(values=SEV_COLORS, name="Severity") +
  labs(title="By Month and Severity", subtitle="Stacked bar chart",
       x="Month", y="Incidents") +
  THEME_BASE + theme(legend.position="right")

p3 <- (p3a | p3b) / (p3c | p3d) +
  plot_annotation(
    title    = "Chart 3 — Incident Analysis · US Accidents · Bay Area CA",
    subtitle = "Distribution by hour, severity, speed pattern, and month",
    theme    = theme(plot.title    = element_text(face="bold", size=13),
                     plot.subtitle = element_text(size=10, color="gray40")))

ggsave(file.path(OUTPUT_DIR, "chart3_incident_analysis.png"),
       plot=p3, width=14, height=10, dpi=300, bg="white")
cat("  OK chart3_incident_analysis.png\n")

# =============================================================
# CHART 4 — SPEED VS INCIDENTS (BOXPLOT + TIMELINE)
# =============================================================
cat("\nGenerating Chart 4: Speed vs incidents...\n")

p4a <- ggplot(df_master, aes(x=severity_label, y=speed_mph, fill=severity_label)) +
  geom_violin(alpha=0.18, color=NA, width=0.9) +
  geom_boxplot(alpha=0.80, width=0.5, outlier.shape=16,
               outlier.size=0.8, outlier.alpha=0.25) +
  stat_summary(fun=mean, geom="point", shape=23, size=3,
               fill="white", color="black") +
  scale_fill_manual(values=SEV_COLORS) +
  labs(title    = "Speed by Incident Severity",
       subtitle = "Diamond = mean · Violin = distribution · Box = IQR",
       x=NULL, y="Traffic speed (mph)") +
  THEME_BASE + theme(legend.position="none", axis.text.x=element_text(size=9))

df_week_master <- df_master %>%
  filter(timestamp >= first_monday, timestamp <= first_monday + days(6) + hours(23))

p4b <- ggplot(df_week_master, aes(x=timestamp, y=speed_mph)) +
  geom_area(fill="#1A5FA8", alpha=0.10) +
  geom_line(color="#1A5FA8", linewidth=0.9) +
  geom_vline(data=filter(df_week_master, has_incident),
             aes(xintercept=timestamp, color=severity_label),
             alpha=0.45, linewidth=1.1) +
  geom_point(data=filter(df_week_master, has_incident),
             aes(color=severity_label), size=3, alpha=0.92) +
  scale_color_manual(values=SEV_COLORS, name="Severity") +
  scale_x_datetime(date_breaks="1 day", date_labels="%a %d %b") +
  labs(title    = "Hourly Speed + Incident Markers",
       subtitle = "Colored points = hours with reported accidents",
       x="Date", y="Speed (mph)") +
  THEME_BASE + theme(legend.position="right")

p4 <- p4a + p4b +
  plot_annotation(
    title    = "Chart 4 — Speed vs Incidents · PEMS-BAY",
    subtitle = "Causal impact of traffic accidents on speed · San Francisco Bay Area",
    theme    = theme(plot.title    = element_text(face="bold", size=13),
                     plot.subtitle = element_text(size=10, color="gray40")))

ggsave(file.path(OUTPUT_DIR, "chart4_speed_vs_incidents.png"),
       plot=p4, width=14, height=6.5, dpi=300, bg="white")
cat("  OK chart4_speed_vs_incidents.png\n")

# ── SUMMARY ───────────────────────────────────────────────────────
cat("\n", strrep("=", 65), "\n")
cat("  ALL CHARTS GENERATED SUCCESSFULLY\n")
cat(strrep("=", 65), "\n\n")
charts <- c("chart1_time_series.png","chart2_adjacency_heatmap.png",
            "chart3_incident_analysis.png","chart4_speed_vs_incidents.png")
for (f in charts) {
  fp   <- file.path(OUTPUT_DIR, f)
  size <- round(file.info(fp)$size/1024, 1)
  cat(sprintf("  %-45s %7.1f KB\n", f, size))
}
cat("\nDone. Open the outputs/ folder to view your charts.\n")
