# Initial Data Exploration


### Data dictionary for Cone Penetration Test use case

- **“sondeernummer”**: unique identification number;

- **“(x,y)”**: coordinates in Lambert 72 projection;

- **“Starthoogte sondering”**: height above sea level of the CPT start
  point;

- **“Einddiepte sondering”**: final depth of the CPT measured relative
  to surface level;

- **“diepte”**: depth relative to surface level in meters

- **“diepte_mtaw”**: height above sea level in meters

- **“qc”**: measurement of the resistance of the soil to the penetration
  of the cone tip. Measurement unit is MPa or psi;

- **“fs”**: measurement of the resistance between the soil and the
  friction sleeve of the CPT tool. Measurement unit is kPa or psi;

- **“Fr”**: friction ratio calculated from qc and fs;

- **“qtn”**: normalized qc accounting for pore water pressure;

- **“Frn”**: normalized friction ratio accounting for pore water
  pressure;

- **“icn”**: ‘soil behaviour type index’ based on normalized qc and rf;

- **“sbt”**: ‘Soil behaviour Type klasse’ based on icn intervals from
  the literature. Integer number between 1 and 9;

- **“ksbt”**: estimated hydraulic conductivity derived from icn through
  empirical correlations;

- **“lithostrat_id”**: lithostratigraphic unit. This is the label we
  want to determine.

- how the model will handle new soil types it was not trained on

- Continuous learning to adapt to new data

- A time series usually have noise denoise

- goal is to find if we can given

- domain knowledge

- How to handle rare classes.

- Distribution

- Apart from CPT data what type of other data/knowledge do the
  geological experts use for soil type classification

- IN the CPT data what signals/features do they look at to be able to
  know a soil type. How do they know that a transition occurs at given
  point , ie do they look at the whole series from soldering id or each
  soil type has special series features that an expert will know by
  looking. Do knew layers soil type tend to have low CPT values etc

- How to handle rare classes, What are this soiltype communicating to a
  geotechnician

- If a good model is found, do VETO have plans/would want this to have a
  continuos learning capability. ie by adding labels of pre labelled
  data.

- Geological domain knowledge from survey data , homogeneous
  signal/heterogeneous signal. Sharp boundarys vs smooth transitions

- Segmentation, multi instance learning

- hidden markov model

- [Introduction to Cone Penetration
  Testing](https://www.ags.org.uk/2022/09/introduction-to-cone-penetration-testing/)

- [Robertson](https://www.cpt-robertson.com/PublicationsPDF/Robertson%20Updated%20SBT%20CGJ%202016.pdf)

- first define soil types using known cone penetration metrics then see
  which soil lithostart correspond to this then do the prediction

### Belgium-specific CPT/Lithostrat References

- **Rogiers et al. 2017 (PLOS ONE)** — shows **automated lithostrat
  mapping from CPT SBT** on a large dataset in **northern Belgium**;
  compares SBT-based modeling vs unsupervised/literature charts and
  reports efficiency + accuracy gains.
  ([PLOS](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0176656))

- **Deckers & Goolaerts 2022 (Geologica Belgica)** — **CPT
  characterization of Middle–Upper Miocene** near **Antwerp Intl.
  Airport** (Berchem Fm: **Kiel & Antwerpen mbrs**); documents typical
  qc/Rf expressions and regional correlations.
  ([popups.uliege.be](https://popups.uliege.be/1374-8505/index.php?id=6999))

- **Schiltz 2020 (Geologica Belgica)** — **NE Belgium (≈60 km²)** case;
  proposes an **informal stratigraphy from CPT signals** and discusses
  which CPT-derived patterns distinguish units.
  ([popups.uliege.be](https://popups.uliege.be/1374-8505/index.php?id=6668&lang=en))

- **DOV Vlaanderen – Sonderingen** — official Flemish **definitions &
  measurement details** (qc, fs, **Rf**, u₂ optional; spacing,
  acquisition); aligns your variables with the **DOV data model**.
  ([Databank Ondergrond
  Vlaanderen](https://www.dov.vlaanderen.be/page/sonderingen))

- **NCS 2022 Excursion Guide (Neogene of N Belgium)** — regional
  context; notes **recent improvements in CPT-based correlations** for
  Antwerp-area units; useful when mapping CPT patterns to named
  formations. ([Databank Ondergrond
  Vlaanderen](https://dov.vlaanderen.be/sites/default/files/pfiles_files/NCS2022_ExcursionGuide.pdf))

- **Robertson CPT guides (SBT/qtn–Fr)** — the **standard
  charts/updates** underpinning SBT interpretation used in Belgian work;
  keep handy for **qₙ, qc, Rf** mapping. ([CPT
  Robertson](https://www.cpt-robertson.com/PublicationsPDF/2-56%20RobSBT.pdf))

- **pydov (DOV client) notebooks** — practical examples to **query CPTs
  by bbox/formation**, fetch profiles, and assemble **median qc/Rf** by
  depth for template building.
  ([Pydov](https://pydov.readthedocs.io/en/stable/notebooks/search_sonderingen.html))

- data management plan

- not all information to start implementing

- relation database; tables

- Preprocessing steps, cleaning, filtering, feature engineering: Talk
  this to the client; works equally good

- minimal viable product (MVP)

- derivative features, interactions, kNN, PCA, tsne, likelihood of
  layers, then which specific layers

- Sliding window approach, Modelling approach, 80% likelihood of this.
  Depth, geographical location, spatial correlation, depth profile,

- sequence, temporal patterns,

- multi instance learning, transition period, transient periods

- predict transition periods

- ensemble models, xgboost, lightgbm, catboost

- missing layers - imputation, knn imputation, mice imputation

- learn characteristics of missing layers

- markov models??

- look at the sequence of layer transitions

- geological domain knowledge, homogeneous signal/heterogeneous signal.
  Sharp boundaries vs smooth transitions

- first maybe group similar layers together, then predict the groups,
  then within the groups predict the specific layers

``` r
library(tidyverse)
library(data.table)
library(arrow)
library(here)
library(xgboost)
library(knitr)
main_folder <- "year2/ProjectDataScience/project"
data_folder <- here(main_folder, "data")
cpt_df <- read_parquet(
  paste(
    data_folder,
    "vw_cpt_brussels_params_completeset_20250318_remapped.parquet",
    sep = "/"
  )
)
setDT(cpt_df)
```

- head

``` r
# filter of na in lithostrat_id
lithostrat_missing_df <- cpt_df[is.na(lithostrat_id)]
cpt_df <- cpt_df[!is.na(lithostrat_id)]
head(cpt_df) %>% kable()
```

| sondering_id | index | pkey_sondering | sondeernummer | x | y | start_sondering_mtaw | diepte_sondering_tot | diepte | diepte_mtaw | qc | fs | qtn | rf | fr | icn | sbt | ksbt | lithostrat_id |
|---:|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 314 | 2593 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 1.6 | 13.66 | 1.17 | 0.035 | 35.89400 | 2.991453 | 3.058371 | 2.564340 | 5 | 1e-07 | Quartair |
| 314 | 2594 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 1.7 | 13.56 | 1.57 | 0.033 | 42.56232 | 2.101911 | 2.138968 | 2.406724 | 5 | 4e-07 | Quartair |
| 314 | 2595 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 1.8 | 13.46 | 1.43 | 0.036 | 38.53699 | 2.517482 | 2.569226 | 2.491219 | 5 | 2e-07 | Quartair |
| 314 | 2596 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 1.9 | 13.36 | 0.50 | 0.024 | 15.67850 | 4.800000 | 5.111166 | 2.982185 | 3 | 0e+00 | Quartair |
| 314 | 2597 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 2.0 | 13.26 | 1.33 | 0.023 | 33.20312 | 1.729323 | 1.772110 | 2.440158 | 5 | 3e-07 | Quartair |
| 314 | 2598 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 153278.2 | 181734.6 | 15.26 | 25.4 | 2.1 | 13.16 | 1.40 | 0.033 | 35.28872 | 2.357143 | 2.415509 | 2.503022 | 5 | 2e-07 | Quartair |

``` r
soldering_id_with_litho_missing <- unique(lithostrat_missing_df$sondering_id)
# without lithostrat_id missing
soldering_id_with_litho <- unique(cpt_df$sondering_id)

id_to_check <- 493
common_soldering_id <- intersect(soldering_id_with_litho_missing, soldering_id_with_litho)
soldering_315_df <- cpt_df[sondering_id == id_to_check]
soldering_315_df_without_litho <- lithostrat_missing_df[sondering_id == id_to_check]
```

``` r
cpt_df[, .N, by = sondering_id][order(-N)]
```

         sondering_id     N
                <int> <int>
      1:         4617  3141
      2:         4647  3013
      3:        14013  2938
      4:        13113  2839
      5:         4643  2787
     ---                   
    238:         3679    46
    239:         3682    33
    240:          493    27
    241:         4873    21
    242:          552    19

``` r
cpt_df[, .(unique_drillings = uniqueN(sondering_id))]
```

       unique_drillings
                  <int>
    1:              242

``` r
# number of obs per soldering id
cpt_df[, no_obs := .N, by = sondering_id]
obs <- cpt_df[, .(
  min_obs = min(no_obs),
  max_obs = max(no_obs),
  mean_obs = mean(no_obs),
  median_obs = median(no_obs)
)]

kable(obs)
```

| min_obs | max_obs | mean_obs | median_obs |
|--------:|--------:|---------:|-----------:|
|      19 |    3141 | 1485.379 |       1479 |

- find unique drillings then map

``` r
unique_drillings <- cpt_df |>
  unique(by = "sondering_id")

## convert lambert to plot to plot with geom_sf
library(sf)
library(ggplot2)
library(ggspatial)
library(ggrepel)


unique_drillings_sf <- st_as_sf(unique_drillings,
  coords = c("x", "y"), crs = 31370
) |>
  st_transform(crs = 4326)

head(unique_drillings_sf) %>% kable()
```

| sondering_id | index | pkey_sondering | sondeernummer | start_sondering_mtaw | diepte_sondering_tot | diepte | diepte_mtaw | qc | fs | qtn | rf | fr | icn | sbt | ksbt | lithostrat_id | no_obs | geometry |
|---:|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|:---|
| 314 | 2593 | https://www.dov.vlaanderen.be/data/sondering/1998-005043 | GEO-97/127-S2 | 15.26 | 25.4 | 1.6 | 13.66 | 1.17 | 0.035 | 35.89400 | 2.9914530 | 3.0583712 | 2.564340 | 5 | 1.00e-07 | Quartair | 239 | POINT (4.4154 50.94589) |
| 315 | 2832 | https://www.dov.vlaanderen.be/data/sondering/1998-005044 | GEO-97/127-S3 | 15.36 | 25.4 | 1.4 | 13.96 | 1.36 | 0.032 | 43.37004 | 2.3529412 | 2.3923445 | 2.432166 | 5 | 4.00e-07 | Quartair | 240 | POINT (4.415635 50.94579) |
| 316 | 3073 | https://www.dov.vlaanderen.be/data/sondering/1998-005045 | GEO-97/127-S4 | 15.44 | 26.4 | 1.1 | 14.34 | 1.12 | 0.050 | 56.62296 | 4.4642857 | 4.5384971 | 2.543799 | 5 | 2.00e-07 | Quartair | 254 | POINT (4.415901 50.94573) |
| 319 | 3327 | https://www.dov.vlaanderen.be/data/sondering/1998-005048 | GEO-97/129-SIII | 17.70 | 14.3 | 1.2 | 16.50 | 5.81 | 0.070 | 146.51291 | 1.2048193 | 1.2088140 | 1.843063 | 6 | 2.23e-05 | Quartair | 132 | POINT (4.411028 50.95289) |
| 374 | 4420 | https://www.dov.vlaanderen.be/data/sondering/1998-005779 | GEO-98/110-S2 | 14.72 | 25.1 | 0.1 | 14.62 | 0.69 | 0.024 | 125.01780 | 3.4782609 | 3.4863451 | 2.234089 | 5 | 1.40e-06 | Quartair | 250 | POINT (4.421244 50.94514) |
| 375 | 4670 | https://www.dov.vlaanderen.be/data/sondering/1998-005780 | GEO-98/110-S3 | 14.90 | 18.4 | 0.1 | 14.80 | 1.79 | 0.016 | 199.00476 | 0.8938547 | 0.8946755 | 1.656611 | 6 | 8.24e-05 | Quartair | 183 | POINT (4.421954 50.94513) |

``` r
# be_lvl2 <- geodata::gadm(
#   country = "BEL",
#   level = 2, path = tempdir()
# ) |>
#   st_as_sf() %>%
#   filter(NAME_2 == "Vlaams Brabant")

# save(be_lvl2, file = paste0(data_folder, "/be_lvl2.rda"))
load(paste0(data_folder, "/be_lvl2.rda"))
ggplot(be_lvl2) +
  geom_sf(fill = "gray") +
  geom_sf(
    data = unique_drillings_sf,
    aes(geometry = geometry),
    color = "blue", size = 0.5
  ) +
  labs(title = "CPT Drillings in Brussels Region") +
  theme_minimal()
```

![](README_files/figure-commonmark/unnamed-chunk-6-1.png)

``` r
library(zoo)

cpt_df[, rf := (fs / (qc * 1000)) * 100]
window_m <- 0.5  # smoothing window in meters
cpt_df[, dz := median(diff(diepte), na.rm = TRUE), 
       by = sondering_id]
cpt_df[(is.na(dz) | !is.finite(dz) | dz <= 0), dz := 0.02]    # fallback 2 cm
cpt_df[(is.na(dz)), dz := 0.02] 
cpt_df[, k := pmax(5L, as.integer(round(window_m / dz))),
       by = sondering_id]
cpt_df[k %% 2 == 0, k := k + 1L]                 # odd window for centering

cpt_df[!is.na(qc), qc_med := rollmedian(qc, k[1], 
                              fill = "extend", 
                              align = "center"), 
       by = sondering_id]
cpt_df[, rf_med := rollmedian(rf, k[1], 
                              fill = "extend", 
                              align = "center"), 
       by = sondering_id]


cpt_df[, cohesive_flag   := as.integer(rf_med >= 1.5)]
cpt_df[, granular_flag   := as.integer(rf_med <= 1.0)]
cpt_df[, organic_flag    := as.integer(qc_med < 0.5 & rf_med > 5)]


cpt_df[, behavior_basic := fcase(
  qc_med < 0.5 & rf_med > 5,
  "Organic/peat",
  qc_med < 1.5 & rf_med >= 2,
  "Soft clay",
  qc_med < 4.0 & rf_med >= 1.5,
  "Clay/Silty clay",
  between(qc_med, 4.0, 10.0)  &
    rf_med <= 1.5,
  "Sand (loose)",
  between(qc_med, 10.0, 20.0) &
    rf_med <= 1.2,
  "Sand (medium)",
  between(qc_med, 20.0, 40.0) &
    rf_med <= 1.0,
  "Sand (dense)",
  qc_med >= 40.0 & rf_med < 0.5,
  "Gravel/Cemented",
  default = "Mixed/Transitional"
)]

cpt_df[, soft_class := fcase(
  qc_med < 1,
  "Very soft",
  between(qc_med, 1, 2),
  "Soft",
  between(qc_med, 2, 4),
  "Firm",
  between(qc_med, 4, 8),
  "Stiff",
  between(qc_med, 8, 16),
  "Very stiff",
  qc_med >= 16,
  "Hard"
)]

cpt_df[, sand_density := fcase(
  granular_flag == 1 & qc_med < 10,
  "Loose",
  granular_flag == 1 &
    between(qc_med, 10, 20),
  "Medium",
  granular_flag == 1 &
    between(qc_med, 20, 40),
  "Dense",
  granular_flag == 1 & qc_med >= 40,
  "Very dense",
  default = NA_character_
)]

cpt_df[, clay_strength := fcase(
  cohesive_flag == 1 & qc_med < 1,
  "Very soft",
  cohesive_flag == 1 &
    between(qc_med, 1, 2),
  "Soft",
  cohesive_flag == 1 &
    between(qc_med, 2, 4),
  "Firm",
  cohesive_flag == 1 &
    between(qc_med, 4, 8),
  "Stiff",
  cohesive_flag == 1 & qc_med >= 8,
  "Very stiff+",
  default = NA_character_
)]
```

``` r
plot_litho_stack <- function(data,
                             litho_col = "lithostrat_id",
                             class_col = "behavior_basic",
                             title = NULL,
                             palette = NULL) {
  stopifnot(all(c(litho_col, class_col) %in% names(data)))

  df <- data[, .N, by = c(litho_col, class_col)]
  df[, prop := N / sum(N), by = litho_col]

  if (is.null(title)) {
    title <- sprintf("Share of %s classes within each %s",
                     class_col, litho_col)
  }

  ggplot(df, aes(x = .data[[litho_col]], y = prop, fill = .data[[class_col]])) +
    geom_col(position = "stack", width = 0.9) +
    scale_y_continuous(labels = scales::percent) +
    scale_fill_manual(values = palette %||% scales::hue_pal()(n_distinct(df[[class_col]]))) +
    labs(
      x = "Lithostratigraphic unit",
      y = "Share (%)",
      fill = class_col,
      title = title
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

plot_litho_stack(cpt_df, class_col = "behavior_basic")
```

![](README_files/figure-commonmark/unnamed-chunk-8-1.png)

``` r
plot_litho_stack(cpt_df, class_col = "soft_class")
```

![](README_files/figure-commonmark/unnamed-chunk-9-1.png)

### What we learned from the literature (CPT → soil subdivision)

- From the literature, we learned that **soil layers can be divided
  using CPT-based soil properties** such as cone resistance (**qc**),
  sleeve friction (**fs**), friction ratio (**Rf = fs/qc**), and
  normalized indices (**Qtn**, **Fr**).
- **Standard CPT charts** (e.g., Robertson) link these properties to
  **soil behavior types (SBT)** like clay, silt, sand, and dense
  sand/gravel.
- Several Belgian studies have shown how CPT data can be used to **map
  lithostratigraphic units more efficiently**:

### Figure: CPT Robertson plots

- The figure below is **derived from CPT measurements**, guided by
  **methods described in the literature**.

- It shows how **different soil types can be distinguished** using
  measured and normalized CPT properties:

  - Left: **qc–Rf space** shows the relation between cone resistance and
    friction ratio.
  - Right: **Qtn–Fr space** shows normalized values useful for comparing
    different sites.

- Colors represent **soil behavior groups** such as *silt/sandy silt*,
  *sand*, *mixed/transitional*, and *dense sand/gravel*.

``` r
library(scales)
library(patchwork)
library(hexbin)


cpt_df[, rf_ratio := fs / qc]
cpt_df[, rf_pct   := 100 * rf_ratio]

# 'fr' appears to be percent already but can have big outliers.
# Make an explicit 'fr_pct' and lightly winsorize for visuals/features.
cpt_df[, fr_pct := as.numeric(fr)]
cpt_df[is.finite(fr_pct), fr_pct := pmin(fr_pct, 20)]   


# summary(cpt_df[, .(qc, fs, rf_ratio, rf_pct, qtn, fr_pct)])


# Prefer normalized Qtn–Fr space if both present; otherwise fall back to qc–Rf.
has_norm <- all(c("qtn", "fr_pct") %in% names(cpt_df))

if (has_norm) {
  # thresholds are sensible starters; tune per basin if desired
  cpt_df[, behavior := fcase(
    fr_pct >= 2   & qtn < 10,                     "Clay / organic",
    fr_pct >= 1   & qtn >= 10 & qtn < 50,         "Silt / sandy silt",
    fr_pct <  1.5 & qtn >= 50 & qtn < 200,        "Sand",
    fr_pct <  0.5 & qtn >= 200,                   "Dense sand / gravel",
    default = "Mixed / transitional"
  )]
} else {
  cpt_df[, behavior := fcase(
    qc < 0.5  & rf_pct > 5,                       "Organic / peat",
    qc < 1.5  & rf_pct >= 2,                       "Clay (soft)",
    qc < 4.0  & rf_pct >= 1.5,                     "Clay / silty clay",
    qc >= 4.0  & qc < 10  & rf_pct <= 1.5,         "Sand (loose)",
    qc >= 10.0 & qc < 20  & rf_pct <= 1.2,         "Sand (medium)",
    qc >= 20.0 & qc < 40  & rf_pct <= 1.0,         "Sand (dense)",
    qc >= 40.0 & rf_pct <  0.5,                    "Dense sand / gravel",
    default = "Mixed / transitional"
  )]
}

cpt_df[, behavior := factor(behavior)]
```

``` r
plot_robertson_pair <- function(dt,
                                sonder_id,
                                color_var = "behavior",
                                palette = "Set2",
                                point_alpha = 0.55,
                                point_size = 1,
                                title_prefix = "CPT Robertson plots") {
  stopifnot(data.table::is.data.table(dt))
  stopifnot(color_var %in% names(dt))

  sub_dt <- dt[sondering_id == sonder_id]
  if (nrow(sub_dt) == 0L) stop("No rows for sondering_id = ", sonder_id)

  col_sym <- rlang::ensym(color_var)

  pA <- ggplot(sub_dt, aes(x = rf_pct, y = qc, color = !!col_sym)) +
    geom_point(alpha = point_alpha, size = point_size) +
    scale_y_log10(name = expression(paste("Cone resistance  ", q[c], " (MPa)")),
                  breaks = c(0.1, 0.5, 1, 5, 10, 50),
                  limits = c(0.1, 60)) +
    scale_x_continuous(name = expression(paste("Friction ratio  ", R[f], " (%)")),
                       limits = c(0, 8),
                       expand = expansion(mult = c(0.01, 0.05))) +
    labs(title = "qc–Rf space", color = color_var) +
    theme_minimal(base_size = 11)

  pB <- ggplot(sub_dt, aes(x = fr_pct, y = qtn, color = !!col_sym)) +
    geom_point(alpha = point_alpha, size = point_size) +
    scale_x_log10(name = expression(paste("Normalized friction ratio  ", F[r], " (%)")),
                  breaks = c(0.1, 0.2, 0.5, 1, 2, 5, 10),
                  limits = c(0.1, 10)) +
    scale_y_log10(name = expression(paste("Normalized cone resistance  ", Q[t][n])),
                  breaks = c(1, 5, 10, 50, 100, 500),
                  limits = c(1, 600)) +
    labs(title = "Qtn–Fr space", color = color_var) +
    theme_minimal(base_size = 11)

  if (!is.null(palette)) {
    pA <- pA + scale_color_brewer(palette = palette)
    pB <- pB + scale_color_brewer(palette = palette)
  }

  (pA | pB) +
    patchwork::plot_annotation(
      title = sprintf("%s — sondering %s", title_prefix, sonder_id)
    )
}


ids <- sample(unique(cpt_df$sondering_id), 10)
for (sid in ids) print(plot_robertson_pair(cpt_df, sid, color_var = "behavior"))
```

![](README_files/figure-commonmark/unnamed-chunk-11-1.png)

![](README_files/figure-commonmark/unnamed-chunk-11-2.png)

![](README_files/figure-commonmark/unnamed-chunk-11-3.png)

![](README_files/figure-commonmark/unnamed-chunk-11-4.png)

![](README_files/figure-commonmark/unnamed-chunk-11-5.png)

![](README_files/figure-commonmark/unnamed-chunk-11-6.png)

![](README_files/figure-commonmark/unnamed-chunk-11-7.png)

![](README_files/figure-commonmark/unnamed-chunk-11-8.png)

![](README_files/figure-commonmark/unnamed-chunk-11-9.png)

![](README_files/figure-commonmark/unnamed-chunk-11-10.png)

- Color the same abobe with lithorstat ID

``` r
for (sid in ids) print(plot_robertson_pair(cpt_df, sid, color_var = "lithostrat_id"))
```

![](README_files/figure-commonmark/unnamed-chunk-12-1.png)

![](README_files/figure-commonmark/unnamed-chunk-12-2.png)

![](README_files/figure-commonmark/unnamed-chunk-12-3.png)

![](README_files/figure-commonmark/unnamed-chunk-12-4.png)

![](README_files/figure-commonmark/unnamed-chunk-12-5.png)

![](README_files/figure-commonmark/unnamed-chunk-12-6.png)

![](README_files/figure-commonmark/unnamed-chunk-12-7.png)

![](README_files/figure-commonmark/unnamed-chunk-12-8.png)

![](README_files/figure-commonmark/unnamed-chunk-12-9.png)

![](README_files/figure-commonmark/unnamed-chunk-12-10.png)

- Randomly select 10 drillings and plot cpt data

``` r
set.seed(124)
sampled_drillings <- sample(unique(cpt_df$sondering_id), 6)
sampled_data <- cpt_df[sondering_id %in% sampled_drillings]
data.table::setorder(sampled_data, index)



plot_cpt_series <- function(data,
                            depth_col = "diepte",
                            value_col = "qc",
                            color_var = "lithostrat_id",
                            group_var = NULL,
                            facet_var = "sondering_id",
                            title = NULL,
                            depth_label = "Depth below surface (m)",
                            value_label = NULL,
                            ncol_facet = 3,
                            log_value = FALSE,
                            reverse_depth = TRUE,
                            flip_coords = TRUE,
                            alpha_line = 0.7,
                            alpha_point = 0.6,
                            point_size = 0.6,
                            legend_position = "bottom") {
  depth_sym <- rlang::ensym(depth_col)
  value_sym <- rlang::ensym(value_col)
  color_sym <- if (is.null(color_var)) {
    NULL
  } else {
    rlang::ensym(color_var)
  }
  group_sym <- if (is.null(group_var) && !is.null(color_var)) {
    rlang::ensym(color_var)
  } else if (is.null(group_var)) {
    rlang::ensym(facet_var)
  } else {
    rlang::ensym(group_var)
  }
  value_label <- value_label %||% value_col

  p <- ggplot(data, aes(x = !!depth_sym, y = !!value_sym)) +
    geom_line(aes(group = interaction(!!rlang::enquo(group_sym))), alpha = alpha_line) +
    geom_point(aes(group = interaction(!!rlang::enquo(group_sym))), size = point_size, alpha = alpha_point)

  if (!is.null(color_sym)) {
    p <- p + aes(color = factor(!!color_sym))
  }

  if (log_value) {
    p <- p + scale_y_log10()
  }
  if (reverse_depth) {
    p <- p + scale_x_reverse()
  }
  if (flip_coords) {
    p <- p + coord_flip()
  }

  p +
    facet_wrap(stats::as.formula(paste("~", facet_var)), ncol = ncol_facet) +
    labs(
      title = title %||% sprintf("CPT Profiles (%s vs depth)", value_col),
      x = depth_label,
      y = value_label,
      color = color_var %||% NULL
    ) +
    theme_minimal() +
    theme(legend.position = legend_position)
}


plot_cpt_series(sampled_data,
  value_col = "qc",
  title = "Cone resistance"
)
```

![](README_files/figure-commonmark/unnamed-chunk-13-1.png)

``` r
plot_cpt_series(sampled_data,
  value_col = "fr",
  title = "friction"
)
```

![](README_files/figure-commonmark/unnamed-chunk-14-1.png)

- lets smooth the qc and fs with rolling mean of window 3
- is the frequency obeying some symmetry

``` r
# use decompose function to
extract_trend <- function(x, freq = 5) {
  decomposed <- decompose(ts(x, frequency = freq))
  return(decomposed$trend)
}

freq_from_depth <- function(z, default = 25L) {
  if (length(z) < 2L) {
    return(default)
  }
  dz <- stats::median(diff(sort(z)), na.rm = TRUE)
  if (!is.finite(dz) || dz <= 0) {
    return(default)
  }
  as.integer(max(1, round(1 / dz)))
}

freq_from_depth <- function(z, default = 25L) {
  if (length(z) < 2L) {
    return(default)
  }
  dz <- stats::median(diff(sort(z)), na.rm = TRUE)
  if (!is.finite(dz) || dz <= 0) {
    return(default)
  }
  as.integer(max(1, round(1 / dz)))
}


sampled_data[, fr_smooth := extract_trend(fr, freq = freq_from_depth(diepte)),
  by = sondering_id
]
sampled_data[, qc_smooth := extract_trend(qc, freq = freq_from_depth(diepte)),
  by = sondering_id
]
```

- plot qc_smooth

``` r
plot_cpt_series(sampled_data,
  value_col = "qc_smooth",
  title = "Smoothed Cone resistance (trend)"
)
```

![](README_files/figure-commonmark/unnamed-chunk-16-1.png)

- plot fr_smooth

``` r
plot_cpt_series(sampled_data,
  value_col = "fr_smooth",
  title = "Smoothed friction (trend)"
)
```

![](README_files/figure-commonmark/unnamed-chunk-17-1.png)

``` r
library(signal)
library(pracma)
# Zero-phase Butterworth low-pass
butter_zero_phase <- function(x, dz, fc = 1.0, order = 4L) {
  # fc in cycles/m; dz in meters/sample
  if (!is.finite(dz) || dz <= 0 || all(!is.finite(x))) return(x)
  nyq <- 1/(2*dz)                    # Nyquist (cycles/m)
  Wc  <- min(0.99, max(1e-6, fc/nyq))# normalized cutoff 0..1
  bf  <- signal::butter(order, Wc, type = "low")
  # detrend first to keep DC drift from dominating:
  xd  <- pracma::detrend(x, tt = "linear")
  as.numeric(signal::filtfilt(bf, xd))
}

vars_to_plot <- c("qc", "fs", "rf_pct", "qtn", "fr_pct", "fr")

filterd_vars <- paste0(vars_to_plot, "butter_zero")
cpt_df[, dz := median(diff(diepte), na.rm = TRUE), by = sondering_id]
cpt_df[, (filterd_vars) := lapply(.SD, butter_zero_phase, dz =dz[1]),
       .SDcols = vars_to_plot,
       by = sondering_id
]
```

``` r
# boxplotqc, rf_pct, qtn, fr by lithostrat_id

vars_to_plot <- c("qc", "fs", "rf_pct", "qtn", "fr_pct", "fr")

lapply(X = vars_to_plot, FUN = function(var) {
    p <- ggplot(cpt_df, aes(x = lithostrat_id, y = log(.data[[var]]), fill = lithostrat_id)) +
        geom_boxplot() +
        scale_y_log10() +
        labs(
        title = paste("Distribution of", var, "Values by Lithostrat Unit"),
        x = "Lithostratigraphic Unit",
        y = paste(var, "[log scale]"),
        fill = "Lithostrat unit"
        ) +
        theme_minimal() +
        theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none"
        )
    p
    
    })
```

    [[1]]

![](README_files/figure-commonmark/unnamed-chunk-20-1.png)


    [[2]]

![](README_files/figure-commonmark/unnamed-chunk-20-2.png)


    [[3]]

![](README_files/figure-commonmark/unnamed-chunk-20-3.png)


    [[4]]

![](README_files/figure-commonmark/unnamed-chunk-20-4.png)


    [[5]]

![](README_files/figure-commonmark/unnamed-chunk-20-5.png)


    [[6]]

![](README_files/figure-commonmark/unnamed-chunk-20-6.png)

``` r
cpt_model_df <- cpt_df[!is.na(qc) & !is.na(lithostrat_id) & !is.na(fs)]

# 1) Five-number summary of qc PER DRILLING (sondering_id)

# cut depth into bins of 1m
cpt_model_df[, diepte := round(diepte)]
cpt_model_df[, diepte_bin := cut(diepte, breaks = seq(
  from = 0, to = max(diepte, na.rm = TRUE), by = 5
))]

summ_by_sonder <- cpt_model_df[
  , .(
    n          = .N,
    depth_min  = min(diepte, na.rm = TRUE),
    depth_max  = max(diepte, na.rm = TRUE),
    qc_min     = min(qc, na.rm = TRUE),
    qc_q25     = quantile(qc, 0.25, na.rm = TRUE),
    qc_med     = median(qc, na.rm = TRUE),
    qc_q75     = quantile(qc, 0.75, na.rm = TRUE),
    qc_max     = max(qc, na.rm = TRUE),
    qc_mean    = mean(qc, na.rm = TRUE),
    qc_sd      = sd(qc, na.rm = TRUE)
  ),
  by = .(sondering_id, diepte_bin, lithostrat_id)
]

library(knitr)
kable(head(summ_by_sonder, 10))
```

| sondering_id | diepte_bin | lithostrat_id | n | depth_min | depth_max | qc_min | qc_q25 | qc_med | qc_q75 | qc_max | qc_mean | qc_sd |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 314 | (0,5\] | Quartair | 39 | 2 | 5 | 0.29 | 0.4400 | 0.670 | 1.5850 | 2.21 | 1.003077 | 0.6506319 |
| 314 | (5,10\] | Quartair | 51 | 6 | 10 | 0.36 | 0.5000 | 1.120 | 7.6200 | 16.71 | 3.940784 | 4.8749637 |
| 314 | (10,15\] | Quartair | 44 | 11 | 15 | 3.21 | 14.8000 | 22.475 | 28.6625 | 40.79 | 21.730909 | 9.5324381 |
| 314 | (10,15\] | Mont_Panisel | 5 | 15 | 15 | 3.19 | 3.2400 | 3.500 | 3.5700 | 6.24 | 3.948000 | 1.2915766 |
| 314 | (15,20\] | Mont_Panisel | 51 | 16 | 20 | 2.38 | 3.5000 | 4.140 | 5.0250 | 13.57 | 4.837255 | 2.2017893 |
| 314 | (20,25\] | Mont_Panisel | 20 | 21 | 22 | 2.31 | 2.9275 | 3.120 | 3.9800 | 5.24 | 3.515000 | 0.8909870 |
| 314 | (20,25\] | Aalbeke | 29 | 23 | 25 | 2.31 | 2.7100 | 2.880 | 3.1700 | 3.69 | 2.928621 | 0.3183969 |
| 315 | (0,5\] | Quartair | 41 | 1 | 5 | 0.50 | 0.6700 | 1.360 | 2.7900 | 4.98 | 1.799268 | 1.3542902 |
| 315 | (5,10\] | Quartair | 51 | 6 | 10 | 0.50 | 0.6700 | 0.860 | 3.4650 | 15.38 | 3.316078 | 4.2564901 |
| 315 | (10,15\] | Quartair | 47 | 11 | 15 | 5.02 | 8.1900 | 11.620 | 13.5000 | 28.14 | 12.141702 | 5.3481680 |

``` r
summ_by_sonder_melt <- melt(
  summ_by_sonder,
  id.vars = c("sondering_id", "diepte_bin", "lithostrat_id"),
  variable.name = "statistic",
  value.name = "value",
  variable.factor = FALSE
)
# remove N

summ_by_sonder_melt <- summ_by_sonder_melt[statistic != "n"]

head(summ_by_sonder_melt)
```

       sondering_id diepte_bin lithostrat_id statistic value
              <int>     <fctr>        <char>    <char> <num>
    1:          314      (0,5]      Quartair depth_min     2
    2:          314     (5,10]      Quartair depth_min     6
    3:          314    (10,15]      Quartair depth_min    11
    4:          314    (10,15]  Mont_Panisel depth_min    15
    5:          314    (15,20]  Mont_Panisel depth_min    16
    6:          314    (20,25]  Mont_Panisel depth_min    21

``` r
## boxplot of qc per lithostrat_id
# ggplot(summ_by_sonder_melt, aes(x = lithostrat_id, y = value, fill = as.factor(diepte))) +
#   geom_boxplot() +
#   scale_y_log10() +
#   labs(
#     title = "Distribution of Cone Resistance (qc) Values",
#     x = "Lithostratigraphic Unit",
#     y = "Cone Resistance (qc) [MPa, log scale]"
#   ) +
#   theme_minimal() +
#   theme(legend.position = "none") +
#   facet_wrap(~ statistic, ncol = 2)

stats_vars <- unique(summ_by_sonder_melt$statistic) %>% sample()
# stats_vars[1:3]
# find unique depth
unique_depths <- unique(summ_by_sonder_melt$diepte)
length(unique_depths)
```

    [1] 9

``` r
for (s in stats_vars) {
  df_s <- summ_by_sonder_melt[statistic == s]

  p <- ggplot(
    df_s,
    aes(x = lithostrat_id, y = value, fill = as.factor(diepte_bin))
  ) +
    geom_boxplot() +
    scale_y_log10() +
    labs(
      title = paste("Distribution of qc by Lithostrat (statistic:", s, ")"),
      x = "Lithostratigraphic Unit",
      y = "Cone Resistance (qc) [MPa, log scale]",
      fill = "Depth bin (m)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )

  print(p)
}
```

![](README_files/figure-commonmark/qc_summaries_simple-1.png)

![](README_files/figure-commonmark/qc_summaries_simple-2.png)

![](README_files/figure-commonmark/qc_summaries_simple-3.png)

![](README_files/figure-commonmark/qc_summaries_simple-4.png)

![](README_files/figure-commonmark/qc_summaries_simple-5.png)

![](README_files/figure-commonmark/qc_summaries_simple-6.png)

![](README_files/figure-commonmark/qc_summaries_simple-7.png)

![](README_files/figure-commonmark/qc_summaries_simple-8.png)

![](README_files/figure-commonmark/qc_summaries_simple-9.png)

``` r
# ..
```

### Scatter of depth vs qc colored by lithostrat_id

``` r
ggplot(cpt_model_df, aes(
  x = diepte,
  y = qc,
  color = as.factor(lithostrat_id)
)) +
  geom_point(alpha = 0.5, size = 0.7) +
  scale_y_log10() +
  scale_x_reverse() +
  coord_flip() +
  labs(
    title = "Scatter Plot of Cone Resistance (qc) vs Depth",
    x = "Depth below surface (m)",
    y = "Cone Resistance (qc) [MPa, log scale]",
    color = "Lithostrat unit"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

![](README_files/figure-commonmark/unnamed-chunk-21-1.png)

``` r
ggplot(cpt_model_df, aes(
  x = diepte,
  y = fr,
  color = as.factor(lithostrat_id)
)) +
  geom_point(alpha = 0.5, size = 0.7) +
  scale_y_log10() +
  scale_x_reverse() +
  coord_flip() +
  labs(
    title = "",
    x = "Depth below surface (m)",
    y = "Cone Resistance (qc) [MPa, log scale]",
    color = "Lithostrat unit"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

![](README_files/figure-commonmark/unnamed-chunk-22-1.png)

``` r
ggplot(cpt_model_df, aes(
  x = diepte,
  y = fs,
  color = as.factor(lithostrat_id)
)) +
  geom_point(alpha = 0.5, size = 0.7) +
  scale_y_log10() +
  scale_x_reverse() +
  coord_flip() +
  labs(
    title = "Scatter Plot of Cone Resistance (fs) vs Depth",
    x = "Depth below surface (m)",
    y = "Cone Resistance (qc) [MPa, log scale]",
    color = "Lithostrat unit"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

![](README_files/figure-commonmark/unnamed-chunk-23-1.png)

- bar plot of lithostrat_id vs depth bins

``` r
# bar plot of lithostrat_id  vs depth bins
ggplot(cpt_model_df, aes(x = diepte_bin, fill = as.factor(lithostrat_id))) +
  geom_bar(position = "fill") +
  labs(
    title = "Proportion of Lithostrat Units by Depth Bins",
    x = "Depth Bins (m)",
    y = "Proportion",
    fill = "Lithostrat unit"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )
```

![](README_files/figure-commonmark/litho_depth_bar-1.png)

``` r
cpt_seq <- cpt_df[, .(col_seq = paste0(rev(unique(lithostrat_id)), 
                                       collapse = "|"),
                      len = uniqueN(lithostrat_id)), 
                  by = sondering_id]

# depth by soldering id
soldering_depth <- cpt_df[, .(min_depth = min(diepte, na.rm = TRUE),
                           max_depth = max(diepte, na.rm = TRUE)),
                       by = sondering_id]
soldering_depth[, depth_range := max_depth - min_depth]

# merge
cpt_seq <- merge(cpt_seq, 
                 soldering_depth, 
                 by = "sondering_id", 
                 all.x = TRUE)
# filter off where len > 1
cpt_seq <- cpt_seq[len > 1]
head(cpt_seq, 20) %>% kable()
```

| sondering_id | col_seq | len | min_depth | max_depth | depth_range |
|---:|:---|---:|---:|---:|---:|
| 314 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 1.60 | 25.40 | 23.80 |
| 315 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 1.40 | 25.30 | 23.90 |
| 316 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 1.10 | 26.40 | 25.30 |
| 374 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 0.10 | 25.00 | 24.90 |
| 375 | Mont_Panisel\|Quartair | 2 | 0.10 | 18.30 | 18.20 |
| 376 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 0.10 | 24.90 | 24.80 |
| 377 | Aalbeke\|Mont_Panisel\|Quartair | 3 | 0.10 | 25.00 | 24.90 |
| 494 | Mons_en_Pevele\|Aalbeke\|Mont_Panisel\|Quartair | 4 | 0.30 | 14.88 | 14.58 |
| 495 | Mons_en_Pevele\|Aalbeke\|Quartair + Mont_Panisel | 3 | 0.10 | 14.88 | 14.78 |
| 496 | Egem\|Gentbrugge\|Quartair | 3 | 0.05 | 15.65 | 15.60 |
| 497 | Egem\|Gentbrugge\|Quartair | 3 | 0.05 | 15.86 | 15.81 |
| 550 | Maldegem\|Quartair | 2 | 1.05 | 11.83 | 10.78 |
| 551 | Maldegem\|Quartair | 2 | 0.05 | 14.17 | 14.12 |
| 553 | Maldegem\|Quartair | 2 | 0.05 | 12.90 | 12.85 |
| 554 | Maldegem\|Quartair | 2 | 0.10 | 12.90 | 12.80 |
| 555 | Maldegem\|Quartair | 2 | 0.15 | 12.95 | 12.80 |
| 1135 | Brussel\|Quartair | 2 | 0.96 | 5.91 | 4.95 |
| 1156 | Brussel\|Quartair | 2 | 0.62 | 12.66 | 12.04 |
| 1158 | Brussel\|Quartair | 2 | 0.04 | 15.10 | 15.06 |
| 1775 | Wemmel\|Asse\|Ursel\|Onbekend | 4 | 1.02 | 20.00 | 18.98 |

``` r
cpt_seq[, layers := strsplit(col_seq, "\\|")]

# Extract all consecutive pairs per sondering
pair_dt <- cpt_seq[, {
  lyr <- unlist(layers)
  if (length(lyr) < 2) {
    data.table(from_layer = character(), to_layer = character())
  } else {
    data.table(
      from_layer = lyr[-length(lyr)],
      to_layer   = lyr[-1]
    )
  }
}, by = sondering_id]

pair_dt[, two_layers := paste(from_layer, to_layer, sep = " -> ")]
kable(pair_dt[1:10])
```

| sondering_id | from_layer   | to_layer     | two_layers                |
|-------------:|:-------------|:-------------|:--------------------------|
|          314 | Aalbeke      | Mont_Panisel | Aalbeke -\> Mont_Panisel  |
|          314 | Mont_Panisel | Quartair     | Mont_Panisel -\> Quartair |
|          315 | Aalbeke      | Mont_Panisel | Aalbeke -\> Mont_Panisel  |
|          315 | Mont_Panisel | Quartair     | Mont_Panisel -\> Quartair |
|          316 | Aalbeke      | Mont_Panisel | Aalbeke -\> Mont_Panisel  |
|          316 | Mont_Panisel | Quartair     | Mont_Panisel -\> Quartair |
|          374 | Aalbeke      | Mont_Panisel | Aalbeke -\> Mont_Panisel  |
|          374 | Mont_Panisel | Quartair     | Mont_Panisel -\> Quartair |
|          375 | Mont_Panisel | Quartair     | Mont_Panisel -\> Quartair |
|          376 | Aalbeke      | Mont_Panisel | Aalbeke -\> Mont_Panisel  |

``` r
#melt to get individual lirthostrat_id

pair_dt_melt <- melt(pair_dt,
                     id.vars = c("sondering_id", "two_layers"),
                     measure.vars = c("from_layer", "to_layer"),
                     variable.name = "layer_type",
                     value.name = "lithostrat_id")
```

``` r
trimmed_mean <- function(x, trim = 0.1, ...) {
  mean(x, trim = trim, ...)
}

dcast_cpt_wide <- function(dt,
                           value_cols,
                           depth_bin = "diepte_bin",
                           id_cols = c("sondering_id", "lithostrat_id"),
                           agg_fun = trimmed_mean,
                           fill_value = NA,
                           na_rm_cols = TRUE) {
  stopifnot(all(value_cols %in% names(dt)))
  wide_list <- lapply(value_cols, function(vcol) {
    myformula <- as.formula(paste(
      paste(id_cols,
        collapse = "+"
      ),
      "~", depth_bin
    ))

    out <- data.table::dcast(
      myformula,
      value.var = vcol,
      fun.aggregate = agg_fun,
      na.rm = TRUE,
      data = dt,
      fill = fill_value
    )
    setnames(
      out, setdiff(names(out), id_cols),
      paste0(setdiff(
        names(out),
        id_cols
      ), "_", vcol)
    )
    out
  })
  wide_dt <- Reduce(function(a, b) {
    merge(a, b, by = id_cols, all = TRUE)
  }, wide_list)
  if (na_rm_cols) {
    keep <- colSums(is.na(wide_dt)) < nrow(wide_dt)
    wide_dt <- wide_dt[, ..keep]
  }
  wide_dt[]
}

# bins of one meter
cpt_model_df <- cpt_model_df[!is.na(diepte)]
cpt_model_df[, diepte_bin := cut(diepte,
  breaks = seq(
    from = 0, to = max(diepte, na.rm = TRUE) + 5, by = 1
  ),
  include.lowest = TRUE
)]


# cpt_model_df[!is.na(fs), fs_smooth := extract_trend(fs, freq = freq_from_depth(diepte)),
#              by = sondering_id]
# cpt_model_df[!is.na(qc), qc_smooth := extract_trend(qc, freq = freq_from_depth(diepte)),
#              by = sondering_id]


value_vars <- c("rf_ratio", "qc", "qtn") # fs", "fr", "icn")

value_vars <- c("rf_pctbutter_zero", "qcbutter_zero", "qtnbutter_zero", "frbutter_zero") 
cpt_model_df1  <- merge(cpt_model_df,
                       pair_dt_melt, 
                       by = c("sondering_id", "lithostrat_id"),
                       all.y = TRUE,
                       allow.cartesian = TRUE)

setorder(cpt_model_df1, sondering_id, -diepte)
cpt_wide <- dcast_cpt_wide(cpt_model_df,
  value_cols = value_vars,
  id_cols = c("sondering_id", "lithostrat_id"),
  fill_value = 0
)
cpt_wide <- cpt_wide[sondering_id %in% pair_dt$sondering_id]
```

``` r
widen_factor_indicators <- function(dt,
                                    factor_cols,
                                    id_cols = c("sondering_id", "lithostrat_id"),
                                    na_label = NULL) {
  stopifnot(data.table::is.data.table(dt))
  stopifnot(all(c(factor_cols, id_cols) %in% names(dt)))

  widen_one <- function(col) {
    tmp <- dt[, c(id_cols, col), with = FALSE]
    setnames(tmp, col, "factor_value")
    if (!is.null(na_label)) tmp[is.na(factor_value), factor_value := na_label]

    wide <- dcast(
      tmp,
      as.formula(paste(paste(id_cols, collapse = " + "), "~ factor_value")),
      value.var = "factor_value",
      fun.aggregate = function(x) as.integer(any(!is.na(x))),
      fill = 0
    )

    value_cols <- setdiff(names(wide), id_cols)
    setnames(wide, value_cols, paste0(col, "_", value_cols))
    wide
  }

  merged <- Reduce(function(a, b) merge(a, b, by = id_cols, all = TRUE),
                   lapply(factor_cols, widen_one))
  num_cols <- setdiff(names(merged), id_cols)
  merged[, (num_cols) := lapply(.SD, function(v) fifelse(is.na(v), 0L, v)), .SDcols = num_cols]
  merged[]
}



factor_vars <- c("behavior", "soft_class")#"soft_class"
cpt_wide_factors <- widen_factor_indicators(cpt_model_df, factor_vars)

cpt_wide <- merge(cpt_wide, cpt_wide_factors,
  by = c("sondering_id", "lithostrat_id"),
  all.x = TRUE
)
```

``` r
ggplot(cpt_wide, aes(x = lithostrat_id, fill = as.factor(lithostrat_id))) +
  geom_bar() +
  labs(
    title = "Bar Plot of Lithostratigraphic Units",
    x = "Lithostratigraphic Unit",
    y = "Count",
    fill = "Lithostratigraphic Unit"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
```

![](README_files/figure-commonmark/unnamed-chunk-29-1.png)

``` r
compress_to_other <- function(dt, col, min_count = 10, new_col = NULL) {
  stopifnot(data.table::is.data.table(dt))
  if (is.null(new_col)) new_col <- paste0(col, "_group")

  counts <- dt[, .N, by = col]
  rare_values <- counts[N < min_count][[col]]

  dt[, (new_col) := get(col)]
  if (length(rare_values)) {
    dt[get(col) %in% rare_values, (new_col) := "Other"]
  }
  dt[, (new_col) := factor(get(new_col))]
  invisible(dt)
}


compress_to_other(cpt_wide,
  col = "lithostrat_id",
  min_count = 10,
  new_col = "lithostrat_id_group"
)

ggplot(cpt_wide, aes(x = lithostrat_id_group, fill = as.factor(lithostrat_id_group))) +
  geom_bar() +
  labs(
    title = "Bar Plot of Lithostratigraphic Units",
    x = "Lithostratigraphic Unit",
    y = "Count",
    fill = "Lithostratigraphic Unit"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
```

``` r
library(xgboost)
library(rsample) 
library(yardstick)

cpt_wide[, lithostrat_id_group := lithostrat_id ]

# 0) Keep only rows with a target
cpt_wide <- cpt_wide[!is.na(lithostrat_id_group)]
#save(cpt_wide, file = here(data_folder, "cpt_wide.rda"))
# 1) Define features (drop ids/target)
feat_cols <- setdiff(names(cpt_wide),
                     c("sondering_id", "lithostrat_id", "lithostrat_id_group", "two_layers"))
#  numeric matrix
X <- as.matrix(cpt_wide[, ..feat_cols])
storage.mode(X) <- "double"

# 2) Encode labels 0..K-1 for xgboost
cpt_wide[, y := as.integer(factor(lithostrat_id_group)) - 1L]
label_levels <- levels(factor(cpt_wide$lithostrat_id_group))
y <- cpt_wide$y
num_class <- length(unique(y))

# 3) Grouped 5-fold CV by sondering_id (sondering_id NOT in the model)
set.seed(42)
gfolds <- group_vfold_cv(
  data = data.frame(id = seq_len(nrow(cpt_wide)),
                    sondering_id = cpt_wide$sondering_id),
  group = sondering_id,
  v = 5
)

oof_prob <- matrix(NA_real_, nrow = nrow(X), ncol = num_class)

# 4) XGBoost params (sensible defaults; tune later)
params <- list(
  objective = "multi:softprob",
  num_class = num_class,
  max_depth = 5,
  eta = 0.05,
  subsample = 0.9,
  colsample_bytree = 0.8,
  eval_metric = "mlogloss",
  tree_method = "hist"
)

for (sp in gfolds$splits) {
  va_idx <- assessment(sp)$id
  tr_idx <- analysis(sp)$id
  dtr <- xgb.DMatrix(X[tr_idx, , drop = FALSE], label = y[tr_idx])
  dva <- xgb.DMatrix(X[va_idx, , drop = FALSE], label = y[va_idx])
  bst <- xgb.train(
    params = params,
    data = dtr,
    nrounds = 400,
    verbose = 0
  )
  oof_prob[va_idx, ] <- matrix(predict(bst, dva), ncol = num_class, byrow = TRUE)
}

# 5) OOF evaluation
oof_pred <- max.col(oof_prob) - 1L
eval_dt <- data.table(
  truth = factor(y, levels = 0:(num_class - 1), labels = label_levels),
  pred  = factor(oof_pred, levels = 0:(num_class - 1), labels = label_levels)
)

metric_set <- yardstick::metric_set(accuracy,
                                    kap, f_meas, 
                                    recall, 
                                    precision)

kable(metric_set(eval_dt,
                 truth = truth, 
                 estimate = pred))
```

| .metric   | .estimator | .estimate |
|:----------|:-----------|----------:|
| accuracy  | multiclass | 0.5786885 |
| kap       | multiclass | 0.5214649 |
| f_meas    | macro      | 0.3922257 |
| recall    | macro      | 0.2489048 |
| precision | macro      | 0.4217692 |

``` r
# Confusion matrix:
mt = conf_mat(eval_dt, 
              truth = truth, 
              estimate = pred)

mt[[1]] |> kable()
```

|  | Aalbeke | Antropogeen | Asse | Bolderberg | Brussel | Diest | Egem | Egemkapel | Gentbrugge | Kwatrecht | Lede | Maldegem | Merelbeke | Mons_en_Pevele | Mont_Panisel | Onbekend | Onbekend + Mont_Panisel | Orchies | Quartair | Quartair + Mont_Panisel | Schelde Groep + Mons_en_Pevele | Sint_Huibrechts_Hern | Tertiair | Tielt | Ursel | Veldwezelt en Gembloux | Vlierzele | Wemmel |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aalbeke | 20 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Antropogeen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 2 | 0 | 19 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 1 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Brussel | 0 | 0 | 0 | 1 | 20 | 1 | 0 | 0 | 0 | 0 | 13 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 2 |
| Diest | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Egem | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Egemkapel | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Gentbrugge | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 1 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| Lede | 0 | 0 | 1 | 1 | 11 | 2 | 0 | 0 | 0 | 0 | 40 | 2 | 0 | 7 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 5 |
| Maldegem | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| Mons_en_Pevele | 3 | 0 | 0 | 0 | 6 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 25 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 |
| Mont_Panisel | 4 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 1 | 2 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Onbekend | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Onbekend + Mont_Panisel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orchies | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Quartair | 3 | 2 | 0 | 3 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 28 | 1 | 0 | 130 | 3 | 0 | 1 | 0 | 0 | 1 | 3 | 0 | 0 |
| Quartair + Mont_Panisel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Schelde Groep + Mons_en_Pevele | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| Tertiair | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tielt | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 2 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 |
| Veldwezelt en Gembloux | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Vlierzele | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 2 | 0 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 30 |

``` r
bst_full <- xgb.train(
  params = params,
  data = xgb.DMatrix(X, label = y),
  nrounds = 400,
  verbose = 0
)

kable(xgb.importance(model = bst_full,
                     feature_names = feat_cols))
```

| Feature                       |      Gain |     Cover | Frequency |
|:------------------------------|----------:|----------:|----------:|
| (2,3\]\_qtnbutter_zero        | 0.0527745 | 0.0165139 | 0.0120581 |
| (1,2\]\_qtnbutter_zero        | 0.0479523 | 0.0216028 | 0.0148952 |
| behavior_Sand                 | 0.0451251 | 0.0326295 | 0.0208151 |
| soft_class_Firm               | 0.0314335 | 0.0231758 | 0.0195330 |
| soft_class_Hard               | 0.0284858 | 0.0176941 | 0.0140495 |
| behavior_Mixed / transitional | 0.0280623 | 0.0181816 | 0.0175415 |
| (3,4\]\_qtnbutter_zero        | 0.0280159 | 0.0098913 | 0.0106667 |
| soft_class_Stiff              | 0.0261342 | 0.0233325 | 0.0225338 |
| soft_class_Very stiff         | 0.0192123 | 0.0144439 | 0.0141587 |
| (1,2\]\_frbutter_zero         | 0.0184853 | 0.0136753 | 0.0127946 |
| (17,18\]\_rf_pctbutter_zero   | 0.0180653 | 0.0112675 | 0.0090299 |
| \[0,1\]\_qtnbutter_zero       | 0.0175473 | 0.0143315 | 0.0110487 |
| soft_class_Soft               | 0.0163389 | 0.0115025 | 0.0171050 |
| (10,11\]\_frbutter_zero       | 0.0151933 | 0.0143763 | 0.0124673 |
| (10,11\]\_rf_pctbutter_zero   | 0.0145352 | 0.0138387 | 0.0143223 |
| behavior_Clay / organic       | 0.0140772 | 0.0101545 | 0.0114033 |
| (6,7\]\_qcbutter_zero         | 0.0135446 | 0.0194971 | 0.0208151 |
| (8,9\]\_rf_pctbutter_zero     | 0.0135253 | 0.0184616 | 0.0159592 |
| (4,5\]\_qcbutter_zero         | 0.0132030 | 0.0136865 | 0.0108304 |
| (7,8\]\_qtnbutter_zero        | 0.0125511 | 0.0125843 | 0.0161502 |
| (14,15\]\_rf_pctbutter_zero   | 0.0120093 | 0.0115410 | 0.0111305 |
| (9,10\]\_qtnbutter_zero       | 0.0117711 | 0.0110648 | 0.0120035 |
| (11,12\]\_qtnbutter_zero      | 0.0115096 | 0.0131034 | 0.0137222 |
| (5,6\]\_qtnbutter_zero        | 0.0113753 | 0.0133577 | 0.0129038 |
| (12,13\]\_qtnbutter_zero      | 0.0111048 | 0.0133323 | 0.0142132 |
| (1,2\]\_qcbutter_zero         | 0.0110344 | 0.0087331 | 0.0110214 |
| (7,8\]\_rf_pctbutter_zero     | 0.0109015 | 0.0107787 | 0.0129310 |
| (12,13\]\_qcbutter_zero       | 0.0108316 | 0.0140624 | 0.0152499 |
| (5,6\]\_rf_pctbutter_zero     | 0.0107272 | 0.0125374 | 0.0149498 |
| (9,10\]\_qcbutter_zero        | 0.0106631 | 0.0114972 | 0.0122217 |
| (9,10\]\_rf_pctbutter_zero    | 0.0102321 | 0.0113318 | 0.0114306 |
| (2,3\]\_qcbutter_zero         | 0.0101114 | 0.0121223 | 0.0163957 |
| (4,5\]\_qtnbutter_zero        | 0.0098533 | 0.0090056 | 0.0086207 |
| (13,14\]\_rf_pctbutter_zero   | 0.0094214 | 0.0090797 | 0.0091936 |
| (13,14\]\_qtnbutter_zero      | 0.0093571 | 0.0144406 | 0.0149225 |
| (5,6\]\_qcbutter_zero         | 0.0086159 | 0.0089394 | 0.0101757 |
| (1,2\]\_rf_pctbutter_zero     | 0.0084848 | 0.0088562 | 0.0102302 |
| (7,8\]\_frbutter_zero         | 0.0083195 | 0.0097891 | 0.0105031 |
| (10,11\]\_qcbutter_zero       | 0.0082609 | 0.0121245 | 0.0124127 |
| behavior_Silt / sandy silt    | 0.0081779 | 0.0106043 | 0.0112942 |
| (17,18\]\_qcbutter_zero       | 0.0081399 | 0.0106440 | 0.0092481 |
| (13,14\]\_qcbutter_zero       | 0.0080828 | 0.0098993 | 0.0145952 |
| (7,8\]\_qcbutter_zero         | 0.0080667 | 0.0116819 | 0.0124127 |
| (3,4\]\_rf_pctbutter_zero     | 0.0080221 | 0.0095026 | 0.0111578 |
| (8,9\]\_qcbutter_zero         | 0.0079875 | 0.0134472 | 0.0140495 |
| (14,15\]\_qcbutter_zero       | 0.0079750 | 0.0089784 | 0.0093027 |
| (2,3\]\_rf_pctbutter_zero     | 0.0079272 | 0.0077038 | 0.0109941 |
| (18,19\]\_qcbutter_zero       | 0.0078169 | 0.0112827 | 0.0099302 |
| (9,10\]\_frbutter_zero        | 0.0078126 | 0.0106284 | 0.0088117 |
| (5,6\]\_frbutter_zero         | 0.0077485 | 0.0097421 | 0.0099847 |
| \[0,1\]\_frbutter_zero        | 0.0076091 | 0.0060222 | 0.0052924 |
| (14,15\]\_frbutter_zero       | 0.0075822 | 0.0083429 | 0.0076113 |
| (12,13\]\_rf_pctbutter_zero   | 0.0075527 | 0.0100660 | 0.0112669 |
| (6,7\]\_qtnbutter_zero        | 0.0073557 | 0.0113389 | 0.0151135 |
| (19,20\]\_qcbutter_zero       | 0.0073135 | 0.0091528 | 0.0076386 |
| (10,11\]\_qtnbutter_zero      | 0.0070273 | 0.0125421 | 0.0146770 |
| (11,12\]\_rf_pctbutter_zero   | 0.0067023 | 0.0091756 | 0.0102030 |
| (3,4\]\_qcbutter_zero         | 0.0062864 | 0.0078084 | 0.0096574 |
| (16,17\]\_rf_pctbutter_zero   | 0.0061356 | 0.0101128 | 0.0103394 |
| (13,14\]\_frbutter_zero       | 0.0060028 | 0.0074370 | 0.0076659 |
| (11,12\]\_qcbutter_zero       | 0.0058856 | 0.0083016 | 0.0098756 |
| (4,5\]\_frbutter_zero         | 0.0058496 | 0.0077207 | 0.0097392 |
| (18,19\]\_rf_pctbutter_zero   | 0.0057078 | 0.0056623 | 0.0055653 |
| (6,7\]\_rf_pctbutter_zero     | 0.0056701 | 0.0065094 | 0.0082660 |
| (4,5\]\_rf_pctbutter_zero     | 0.0055846 | 0.0080986 | 0.0081296 |
| (19,20\]\_rf_pctbutter_zero   | 0.0055733 | 0.0051480 | 0.0045286 |
| (14,15\]\_qtnbutter_zero      | 0.0055440 | 0.0058351 | 0.0068747 |
| (22,23\]\_rf_pctbutter_zero   | 0.0052477 | 0.0027041 | 0.0014732 |
| (16,17\]\_qtnbutter_zero      | 0.0051327 | 0.0068781 | 0.0078568 |
| (8,9\]\_qtnbutter_zero        | 0.0050906 | 0.0071802 | 0.0087025 |
| \[0,1\]\_rf_pctbutter_zero    | 0.0050266 | 0.0060738 | 0.0065746 |
| (6,7\]\_frbutter_zero         | 0.0050062 | 0.0077961 | 0.0085661 |
| (15,16\]\_qtnbutter_zero      | 0.0049580 | 0.0068723 | 0.0062746 |
| (8,9\]\_frbutter_zero         | 0.0047890 | 0.0069039 | 0.0076386 |
| (23,24\]\_qcbutter_zero       | 0.0043859 | 0.0070674 | 0.0042285 |
| \[0,1\]\_qcbutter_zero        | 0.0041138 | 0.0048822 | 0.0060290 |
| (15,16\]\_rf_pctbutter_zero   | 0.0040781 | 0.0059206 | 0.0061654 |
| (11,12\]\_frbutter_zero       | 0.0040593 | 0.0054000 | 0.0066019 |
| (19,20\]\_qtnbutter_zero      | 0.0040353 | 0.0048120 | 0.0042558 |
| behavior_Dense sand / gravel  | 0.0039946 | 0.0058665 | 0.0055925 |
| (15,16\]\_qcbutter_zero       | 0.0039521 | 0.0065354 | 0.0058926 |
| (3,4\]\_frbutter_zero         | 0.0035625 | 0.0046513 | 0.0058653 |
| soft_class_Very soft          | 0.0035620 | 0.0045269 | 0.0061109 |
| (12,13\]\_frbutter_zero       | 0.0035062 | 0.0056440 | 0.0066565 |
| (16,17\]\_qcbutter_zero       | 0.0034057 | 0.0077288 | 0.0084024 |
| (20,21\]\_qcbutter_zero       | 0.0033174 | 0.0047550 | 0.0040921 |
| (15,16\]\_frbutter_zero       | 0.0031831 | 0.0054768 | 0.0050196 |
| (17,18\]\_qtnbutter_zero      | 0.0031508 | 0.0036673 | 0.0039830 |
| (26,27\]\_qtnbutter_zero      | 0.0030984 | 0.0042789 | 0.0025917 |
| (17,18\]\_frbutter_zero       | 0.0029896 | 0.0055372 | 0.0051833 |
| (18,19\]\_qtnbutter_zero      | 0.0029682 | 0.0057795 | 0.0055925 |
| (18,19\]\_frbutter_zero       | 0.0026863 | 0.0046456 | 0.0041194 |
| (2,3\]\_frbutter_zero         | 0.0025774 | 0.0041879 | 0.0067110 |
| (16,17\]\_frbutter_zero       | 0.0025153 | 0.0043166 | 0.0044195 |
| (25,26\]\_rf_pctbutter_zero   | 0.0023518 | 0.0018225 | 0.0007366 |
| (22,23\]\_frbutter_zero       | 0.0023031 | 0.0020959 | 0.0012549 |
| (24,25\]\_qcbutter_zero       | 0.0017936 | 0.0027304 | 0.0026189 |
| (25,26\]\_qtnbutter_zero      | 0.0017138 | 0.0009919 | 0.0009003 |
| (19,20\]\_frbutter_zero       | 0.0016657 | 0.0033227 | 0.0028372 |
| (25,26\]\_qcbutter_zero       | 0.0015227 | 0.0012596 | 0.0014186 |
| (20,21\]\_rf_pctbutter_zero   | 0.0012763 | 0.0038921 | 0.0023189 |
| (28,29\]\_qcbutter_zero       | 0.0011863 | 0.0019371 | 0.0019369 |
| (21,22\]\_frbutter_zero       | 0.0011717 | 0.0033733 | 0.0024553 |
| (29,30\]\_qtnbutter_zero      | 0.0011225 | 0.0018928 | 0.0007911 |
| (22,23\]\_qcbutter_zero       | 0.0010632 | 0.0019913 | 0.0016641 |
| (23,24\]\_rf_pctbutter_zero   | 0.0009548 | 0.0009158 | 0.0007366 |
| (20,21\]\_qtnbutter_zero      | 0.0009510 | 0.0024948 | 0.0026462 |
| (21,22\]\_rf_pctbutter_zero   | 0.0009174 | 0.0008726 | 0.0008184 |
| (24,25\]\_qtnbutter_zero      | 0.0009099 | 0.0012413 | 0.0015004 |
| (24,25\]\_frbutter_zero       | 0.0007495 | 0.0012306 | 0.0013368 |
| (29,30\]\_frbutter_zero       | 0.0007461 | 0.0013846 | 0.0012003 |
| (23,24\]\_qtnbutter_zero      | 0.0007284 | 0.0015249 | 0.0009821 |
| (29,30\]\_qcbutter_zero       | 0.0007262 | 0.0016346 | 0.0009548 |
| (38,39\]\_qtnbutter_zero      | 0.0006519 | 0.0017040 | 0.0006002 |
| (20,21\]\_frbutter_zero       | 0.0006011 | 0.0009048 | 0.0008457 |
| (21,22\]\_qtnbutter_zero      | 0.0005595 | 0.0006754 | 0.0005183 |
| (26,27\]\_qcbutter_zero       | 0.0004259 | 0.0016846 | 0.0012822 |
| (24,25\]\_rf_pctbutter_zero   | 0.0004131 | 0.0011700 | 0.0009548 |
| (27,28\]\_qcbutter_zero       | 0.0003857 | 0.0011141 | 0.0012276 |
| (33,34\]\_qcbutter_zero       | 0.0002680 | 0.0003654 | 0.0001091 |
| (26,27\]\_rf_pctbutter_zero   | 0.0002628 | 0.0007082 | 0.0005729 |
| (33,34\]\_frbutter_zero       | 0.0002604 | 0.0009267 | 0.0003546 |
| (22,23\]\_qtnbutter_zero      | 0.0002479 | 0.0002955 | 0.0002182 |
| (30,31\]\_qcbutter_zero       | 0.0002014 | 0.0001799 | 0.0001364 |
| (21,22\]\_qcbutter_zero       | 0.0001499 | 0.0003095 | 0.0002728 |
| (23,24\]\_frbutter_zero       | 0.0001479 | 0.0004255 | 0.0002728 |
| (38,39\]\_qcbutter_zero       | 0.0001475 | 0.0004287 | 0.0001364 |
| (28,29\]\_rf_pctbutter_zero   | 0.0001172 | 0.0000812 | 0.0001091 |
| (27,28\]\_qtnbutter_zero      | 0.0001163 | 0.0002545 | 0.0004911 |
| (25,26\]\_frbutter_zero       | 0.0001088 | 0.0001528 | 0.0001637 |
| (28,29\]\_frbutter_zero       | 0.0001042 | 0.0001962 | 0.0001910 |
| (33,34\]\_qtnbutter_zero      | 0.0000952 | 0.0000693 | 0.0000273 |
| (27,28\]\_rf_pctbutter_zero   | 0.0000947 | 0.0003289 | 0.0003001 |
| (35,36\]\_qcbutter_zero       | 0.0000915 | 0.0003692 | 0.0001364 |
| (35,36\]\_qtnbutter_zero      | 0.0000803 | 0.0002802 | 0.0001091 |
| (29,30\]\_rf_pctbutter_zero   | 0.0000641 | 0.0000700 | 0.0001364 |
| (33,34\]\_rf_pctbutter_zero   | 0.0000586 | 0.0002465 | 0.0001091 |
| (27,28\]\_frbutter_zero       | 0.0000486 | 0.0001914 | 0.0002182 |
| (26,27\]\_frbutter_zero       | 0.0000414 | 0.0001862 | 0.0001091 |
| (30,31\]\_qtnbutter_zero      | 0.0000225 | 0.0000567 | 0.0000546 |

``` r
setDT(eval_dt)

cm_counts <- dcast(eval_dt[, .N, 
                           by = .(truth, pred)], 
                   truth ~ pred, value.var = "N", 
                   fill = 0)
kable(cm_counts)
```

| truth | Aalbeke | Asse | Bolderberg | Brussel | Egem | Egemkapel | Kwatrecht | Lede | Merelbeke | Mons_en_Pevele | Mont_Panisel | Onbekend | Quartair | Sint_Huibrechts_Hern | Tielt | Ursel | Veldwezelt en Gembloux | Wemmel |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aalbeke | 20 | 2 | 0 | 0 | 0 | 1 | 3 | 0 | 2 | 3 | 4 | 0 | 3 | 0 | 0 | 2 | 0 | 1 |
| Antropogeen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| Asse | 5 | 19 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 1 | 0 | 0 |
| Brussel | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 11 | 0 | 6 | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 2 |
| Diest | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| Egem | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Egemkapel | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Gentbrugge | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Kwatrecht | 5 | 3 | 0 | 0 | 0 | 0 | 8 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| Lede | 0 | 0 | 2 | 13 | 0 | 0 | 0 | 40 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| Maldegem | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Merelbeke | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| Mons_en_Pevele | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 7 | 0 | 25 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| Mont_Panisel | 2 | 1 | 0 | 1 | 0 | 0 | 4 | 3 | 0 | 1 | 31 | 0 | 1 | 0 | 0 | 1 | 0 | 6 |
| Onbekend | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 28 | 0 | 0 | 0 | 0 | 0 |
| Onbekend + Mont_Panisel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Orchies | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Quartair | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 130 | 0 | 0 | 0 | 0 | 0 |
| Quartair + Mont_Panisel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Schelde Groep + Mons_en_Pevele | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 3 |
| Tertiair | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tielt | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 1 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 12 | 0 | 0 |
| Veldwezelt en Gembloux | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Vlierzele | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 5 | 0 | 4 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 30 |

``` r
pred_cols <- setdiff(names(cm_counts), "truth")
cm_counts[, row_total := rowSums(.SD), .SDcols = pred_cols]
cm_props <- copy(cm_counts)
cm_props[, row_total := rowSums(.SD), .SDcols = pred_cols]
cm_props[, (pred_cols) := lapply(.SD, function(x)
  ifelse(row_total > 0, x / row_total, NA_real_)), .SDcols = pred_cols]

kable(cm_props)
```

| truth | Aalbeke | Asse | Bolderberg | Brussel | Egem | Egemkapel | Kwatrecht | Lede | Merelbeke | Mons_en_Pevele | Mont_Panisel | Onbekend | Quartair | Sint_Huibrechts_Hern | Tielt | Ursel | Veldwezelt en Gembloux | Wemmel | row_total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aalbeke | 0.4878049 | 0.0487805 | 0.000000 | 0.0000000 | 0.0 | 0.0243902 | 0.0731707 | 0.0000000 | 0.0487805 | 0.0731707 | 0.0975610 | 0.0000000 | 0.0731707 | 0.0000000 | 0.0000000 | 0.0487805 | 0.000000 | 0.0243902 | 41 |
| Antropogeen | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 2 |
| Asse | 0.1724138 | 0.6551724 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0344828 | 0.0344828 | 0.0000000 | 0.0000000 | 0.0344828 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0689655 | 0.000000 | 0.0000000 | 29 |
| Bolderberg | 0.0000000 | 0.0000000 | 0.000000 | 0.1428571 | 0.0 | 0.0000000 | 0.0000000 | 0.1428571 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.4285714 | 0.1428571 | 0.0000000 | 0.1428571 | 0.000000 | 0.0000000 | 7 |
| Brussel | 0.0000000 | 0.0000000 | 0.000000 | 0.4444444 | 0.0 | 0.0000000 | 0.0000000 | 0.2444444 | 0.0000000 | 0.1333333 | 0.0000000 | 0.0000000 | 0.1111111 | 0.0000000 | 0.0222222 | 0.0000000 | 0.000000 | 0.0444444 | 45 |
| Diest | 0.0000000 | 0.0000000 | 0.000000 | 0.1666667 | 0.0 | 0.0000000 | 0.0000000 | 0.3333333 | 0.0000000 | 0.0000000 | 0.0000000 | 0.1666667 | 0.3333333 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 6 |
| Egem | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.5 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.5000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 2 |
| Egemkapel | 0.0000000 | 1.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Gentbrugge | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 1.0000000 | 0.000000 | 0.0000000 | 2 |
| Kwatrecht | 0.2083333 | 0.1250000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.3333333 | 0.0000000 | 0.0833333 | 0.0000000 | 0.0833333 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.1250000 | 0.000000 | 0.0416667 | 24 |
| Lede | 0.0000000 | 0.0000000 | 0.031746 | 0.2063492 | 0.0 | 0.0000000 | 0.0000000 | 0.6349206 | 0.0000000 | 0.0476190 | 0.0000000 | 0.0158730 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.015873 | 0.0476190 | 63 |
| Maldegem | 0.0000000 | 0.0000000 | 0.000000 | 0.1666667 | 0.0 | 0.0000000 | 0.0000000 | 0.3333333 | 0.0000000 | 0.0000000 | 0.1666667 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.3333333 | 6 |
| Merelbeke | 0.2000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.1000000 | 0.0000000 | 0.1000000 | 0.0000000 | 0.1000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.5000000 | 0.000000 | 0.0000000 | 10 |
| Mons_en_Pevele | 0.0000000 | 0.0217391 | 0.000000 | 0.1086957 | 0.0 | 0.0000000 | 0.0000000 | 0.1521739 | 0.0000000 | 0.5434783 | 0.0434783 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.1304348 | 46 |
| Mont_Panisel | 0.0392157 | 0.0196078 | 0.000000 | 0.0196078 | 0.0 | 0.0000000 | 0.0784314 | 0.0588235 | 0.0000000 | 0.0196078 | 0.6078431 | 0.0000000 | 0.0196078 | 0.0000000 | 0.0000000 | 0.0196078 | 0.000000 | 0.1176471 | 51 |
| Onbekend | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.3333333 | 0.6666667 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 42 |
| Onbekend + Mont_Panisel | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Orchies | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Quartair | 0.0000000 | 0.0069930 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0839161 | 0.9090909 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 143 |
| Quartair + Mont_Panisel | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 3 |
| Schelde Groep + Mons_en_Pevele | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Sint_Huibrechts_Hern | 0.0909091 | 0.1818182 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.1818182 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0909091 | 0.1818182 | 0.0000000 | 0.0000000 | 0.000000 | 0.2727273 | 11 |
| Tertiair | 0.0000000 | 0.0000000 | 0.000000 | 1.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Tielt | 0.0000000 | 0.0000000 | 0.000000 | 0.5000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.5000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 2 |
| Ursel | 0.0476190 | 0.0952381 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.1428571 | 0.0000000 | 0.0952381 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0476190 | 0.0000000 | 0.0000000 | 0.5714286 | 0.000000 | 0.0000000 | 21 |
| Veldwezelt en Gembloux | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 1.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 3 |
| Vlierzele | 0.0000000 | 0.0000000 | 1.000000 | 0.0000000 | 0.0 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.000000 | 0.0000000 | 1 |
| Wemmel | 0.0222222 | 0.0222222 | 0.000000 | 0.0444444 | 0.0 | 0.0000000 | 0.0000000 | 0.1111111 | 0.0000000 | 0.0888889 | 0.0222222 | 0.0000000 | 0.0000000 | 0.0222222 | 0.0000000 | 0.0000000 | 0.000000 | 0.6666667 | 45 |

``` r
# melt by truth and row totals then filter
cmprops_melt <- melt(cm_props,
  id.vars = c("truth", "row_total"),
  variable.name = "pred",
  value.name = "prop"
)

# melt cm_counts by truth and row totals then ,erge with cmprops_melt
cmcounts_melt <- melt(cm_counts,
  id.vars = c("truth", "row_total"),
  variable.name = "pred",
  value.name = "count"
)

cm_melt <- merge(cmprops_melt, cmcounts_melt,
  by = c("truth", "pred", "row_total"),
  all.x = TRUE
)
# prop != 0
cm_melt <- cm_melt[!is.na(prop) & prop > 0]
kable(cm_melt)
```

| truth | pred | row_total | prop | count |
|:---|:---|---:|---:|---:|
| Aalbeke | Aalbeke | 41 | 0.4878049 | 20 |
| Aalbeke | Asse | 41 | 0.0487805 | 2 |
| Aalbeke | Egemkapel | 41 | 0.0243902 | 1 |
| Aalbeke | Kwatrecht | 41 | 0.0731707 | 3 |
| Aalbeke | Merelbeke | 41 | 0.0487805 | 2 |
| Aalbeke | Mons_en_Pevele | 41 | 0.0731707 | 3 |
| Aalbeke | Mont_Panisel | 41 | 0.0975610 | 4 |
| Aalbeke | Quartair | 41 | 0.0731707 | 3 |
| Aalbeke | Ursel | 41 | 0.0487805 | 2 |
| Aalbeke | Wemmel | 41 | 0.0243902 | 1 |
| Antropogeen | Quartair | 2 | 1.0000000 | 2 |
| Asse | Aalbeke | 29 | 0.1724138 | 5 |
| Asse | Asse | 29 | 0.6551724 | 19 |
| Asse | Kwatrecht | 29 | 0.0344828 | 1 |
| Asse | Lede | 29 | 0.0344828 | 1 |
| Asse | Mont_Panisel | 29 | 0.0344828 | 1 |
| Asse | Ursel | 29 | 0.0689655 | 2 |
| Bolderberg | Brussel | 7 | 0.1428571 | 1 |
| Bolderberg | Lede | 7 | 0.1428571 | 1 |
| Bolderberg | Quartair | 7 | 0.4285714 | 3 |
| Bolderberg | Sint_Huibrechts_Hern | 7 | 0.1428571 | 1 |
| Bolderberg | Ursel | 7 | 0.1428571 | 1 |
| Brussel | Brussel | 45 | 0.4444444 | 20 |
| Brussel | Lede | 45 | 0.2444444 | 11 |
| Brussel | Mons_en_Pevele | 45 | 0.1333333 | 6 |
| Brussel | Quartair | 45 | 0.1111111 | 5 |
| Brussel | Tielt | 45 | 0.0222222 | 1 |
| Brussel | Wemmel | 45 | 0.0444444 | 2 |
| Diest | Brussel | 6 | 0.1666667 | 1 |
| Diest | Lede | 6 | 0.3333333 | 2 |
| Diest | Onbekend | 6 | 0.1666667 | 1 |
| Diest | Quartair | 6 | 0.3333333 | 2 |
| Egem | Egem | 2 | 0.5000000 | 1 |
| Egem | Mons_en_Pevele | 2 | 0.5000000 | 1 |
| Egemkapel | Asse | 1 | 1.0000000 | 1 |
| Gentbrugge | Ursel | 2 | 1.0000000 | 2 |
| Kwatrecht | Aalbeke | 24 | 0.2083333 | 5 |
| Kwatrecht | Asse | 24 | 0.1250000 | 3 |
| Kwatrecht | Kwatrecht | 24 | 0.3333333 | 8 |
| Kwatrecht | Merelbeke | 24 | 0.0833333 | 2 |
| Kwatrecht | Mont_Panisel | 24 | 0.0833333 | 2 |
| Kwatrecht | Ursel | 24 | 0.1250000 | 3 |
| Kwatrecht | Wemmel | 24 | 0.0416667 | 1 |
| Lede | Bolderberg | 63 | 0.0317460 | 2 |
| Lede | Brussel | 63 | 0.2063492 | 13 |
| Lede | Lede | 63 | 0.6349206 | 40 |
| Lede | Mons_en_Pevele | 63 | 0.0476190 | 3 |
| Lede | Onbekend | 63 | 0.0158730 | 1 |
| Lede | Veldwezelt en Gembloux | 63 | 0.0158730 | 1 |
| Lede | Wemmel | 63 | 0.0476190 | 3 |
| Maldegem | Brussel | 6 | 0.1666667 | 1 |
| Maldegem | Lede | 6 | 0.3333333 | 2 |
| Maldegem | Mont_Panisel | 6 | 0.1666667 | 1 |
| Maldegem | Wemmel | 6 | 0.3333333 | 2 |
| Merelbeke | Aalbeke | 10 | 0.2000000 | 2 |
| Merelbeke | Kwatrecht | 10 | 0.1000000 | 1 |
| Merelbeke | Merelbeke | 10 | 0.1000000 | 1 |
| Merelbeke | Mont_Panisel | 10 | 0.1000000 | 1 |
| Merelbeke | Ursel | 10 | 0.5000000 | 5 |
| Mons_en_Pevele | Asse | 46 | 0.0217391 | 1 |
| Mons_en_Pevele | Brussel | 46 | 0.1086957 | 5 |
| Mons_en_Pevele | Lede | 46 | 0.1521739 | 7 |
| Mons_en_Pevele | Mons_en_Pevele | 46 | 0.5434783 | 25 |
| Mons_en_Pevele | Mont_Panisel | 46 | 0.0434783 | 2 |
| Mons_en_Pevele | Wemmel | 46 | 0.1304348 | 6 |
| Mont_Panisel | Aalbeke | 51 | 0.0392157 | 2 |
| Mont_Panisel | Asse | 51 | 0.0196078 | 1 |
| Mont_Panisel | Brussel | 51 | 0.0196078 | 1 |
| Mont_Panisel | Kwatrecht | 51 | 0.0784314 | 4 |
| Mont_Panisel | Lede | 51 | 0.0588235 | 3 |
| Mont_Panisel | Mons_en_Pevele | 51 | 0.0196078 | 1 |
| Mont_Panisel | Mont_Panisel | 51 | 0.6078431 | 31 |
| Mont_Panisel | Quartair | 51 | 0.0196078 | 1 |
| Mont_Panisel | Ursel | 51 | 0.0196078 | 1 |
| Mont_Panisel | Wemmel | 51 | 0.1176471 | 6 |
| Onbekend | Onbekend | 42 | 0.3333333 | 14 |
| Onbekend | Quartair | 42 | 0.6666667 | 28 |
| Onbekend + Mont_Panisel | Quartair | 1 | 1.0000000 | 1 |
| Orchies | Kwatrecht | 1 | 1.0000000 | 1 |
| Quartair | Asse | 143 | 0.0069930 | 1 |
| Quartair | Onbekend | 143 | 0.0839161 | 12 |
| Quartair | Quartair | 143 | 0.9090909 | 130 |
| Quartair + Mont_Panisel | Quartair | 3 | 1.0000000 | 3 |
| Schelde Groep + Mons_en_Pevele | Lede | 1 | 1.0000000 | 1 |
| Sint_Huibrechts_Hern | Aalbeke | 11 | 0.0909091 | 1 |
| Sint_Huibrechts_Hern | Asse | 11 | 0.1818182 | 2 |
| Sint_Huibrechts_Hern | Lede | 11 | 0.1818182 | 2 |
| Sint_Huibrechts_Hern | Quartair | 11 | 0.0909091 | 1 |
| Sint_Huibrechts_Hern | Sint_Huibrechts_Hern | 11 | 0.1818182 | 2 |
| Sint_Huibrechts_Hern | Wemmel | 11 | 0.2727273 | 3 |
| Tertiair | Brussel | 1 | 1.0000000 | 1 |
| Tielt | Brussel | 2 | 0.5000000 | 1 |
| Tielt | Mons_en_Pevele | 2 | 0.5000000 | 1 |
| Ursel | Aalbeke | 21 | 0.0476190 | 1 |
| Ursel | Asse | 21 | 0.0952381 | 2 |
| Ursel | Kwatrecht | 21 | 0.1428571 | 3 |
| Ursel | Merelbeke | 21 | 0.0952381 | 2 |
| Ursel | Quartair | 21 | 0.0476190 | 1 |
| Ursel | Ursel | 21 | 0.5714286 | 12 |
| Veldwezelt en Gembloux | Quartair | 3 | 1.0000000 | 3 |
| Vlierzele | Bolderberg | 1 | 1.0000000 | 1 |
| Wemmel | Aalbeke | 45 | 0.0222222 | 1 |
| Wemmel | Asse | 45 | 0.0222222 | 1 |
| Wemmel | Brussel | 45 | 0.0444444 | 2 |
| Wemmel | Lede | 45 | 0.1111111 | 5 |
| Wemmel | Mons_en_Pevele | 45 | 0.0888889 | 4 |
| Wemmel | Mont_Panisel | 45 | 0.0222222 | 1 |
| Wemmel | Sint_Huibrechts_Hern | 45 | 0.0222222 | 1 |
| Wemmel | Wemmel | 45 | 0.6666667 | 30 |

``` r
library(tidymodels)
library(workflowsets)
library(doParallel)

set.seed(42)

# 
# Ensure outcome is a factor

# remove soil type with less than 5 samples

litho_n <- cpt_wide[, .N, by = lithostrat_id]
rare_litho <- litho_n[N < 5, lithostrat_id]
cpt_wide <- cpt_wide[!lithostrat_id %in% rare_litho]
cpt_wide <- cpt_wide[, lithostrat_id := as.factor(lithostrat_id)]

# 
id_cols <- c(
             "lithostrat_id", "lithostrat_id_group",
             "two_layers", "y")             

# come up with x cols

x_cols <- setdiff(names(cpt_wide),id_cols)

## add back ticks 
x_cols_quoted <- paste0("`", x_cols, "`")
formula <- paste("lithostrat_id ~", 
    paste(x_cols_quoted, collapse = " + ")) %>%
  as.formula()


# Grouped CV (no stratification in group_vfold_cv) 

gfolds <- rsample::group_vfold_cv(cpt_wide, 
                                  group = sondering_id, 
                                  v = 5)




# Trees don't need scaling, but step_zv helps remove single-valued cols
rec <- recipe(formula, data = cpt_wide) |>
  update_role(sondering_id, new_role = "group_id") |>
  step_rm(sondering_id) |>
  step_zv()

# Model specs
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) |>
  set_mode("classification") |>
  set_engine("ranger", importance = "impurity", probability = TRUE)

xgb_spec <- boost_tree(
  trees          = 1000,
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  min_n          = tune(),
  mtry           = tune(),
  sample_size    = tune()
) |>
  set_mode("classification") |>
  set_engine("xgboost")

# Build workflows FIRST (required to extract parameter sets)
wf_rf  <- workflow() |> add_recipe(rec) |> add_model(rf_spec)
wf_xgb <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)

# Extract tunable parameters from workflows
rf_params  <- extract_parameter_set_dials(wf_rf)
xgb_params <- extract_parameter_set_dials(wf_xgb)

# Finalize unknown parameters (mtry, sample_size) using the training data
rf_params  <- finalize(rf_params,  cpt_wide[, ..x_cols])
xgb_params <- finalize(xgb_params, cpt_wide[, ..x_cols])

# Generate grids
rf_grid  <- grid_latin_hypercube(rf_params,  size = 20)
xgb_grid <- grid_latin_hypercube(xgb_params, size = 30)

# Workflow set (for parallel tuning)
wfs <- workflow_set(
  preproc = list(base = rec),
  models  = list(rf = rf_spec, xgb = xgb_spec)
)

multi_metrics <- yardstick::metric_set(
  accuracy, 
  bal_accuracy,
  kap)


ctrl <- control_grid(
  save_pred = TRUE, save_workflow = TRUE,
  parallel_over = "resamples", allow_par = TRUE, verbose = TRUE
)

library(future)
library(furrr)

# Set up parallel backend (leave 4 cores free)
plan(multisession, workers = parallel::detectCores() - 4)
# Tune both models on the same resamples ---
# Supply per-model grids using a named list that matches wfs$model labels
# Tune RF
rf_res <- tune_grid(
  wf_rf,
  resamples = gfolds,
  grid      = rf_grid,
  metrics   = multi_metrics,
  control   = ctrl
)

# Tune XGBoost
xgb_res <- tune_grid(
  wf_xgb,
  resamples = gfolds,
  grid      = xgb_grid,
  metrics   = multi_metrics,
  control   = ctrl
)

plan(sequential) # back to sequential
# Best configs
best_rf  <- select_best(rf_res,  metric = "accuracy")
best_xgb <- select_best(xgb_res, metric = "accuracy")

# Finalize workflows
wf_rf_final  <- finalize_workflow(wf_rf,  best_rf)
wf_xgb_final <- finalize_workflow(wf_xgb, best_xgb)

# Refit for OOF predictions
ctrl_oof <- control_resamples(save_pred = TRUE)

rf_oof  <- fit_resamples(wf_rf_final,  resamples = gfolds, metrics = multi_metrics, control = ctrl_oof)
xgb_oof <- fit_resamples(wf_xgb_final, resamples = gfolds, metrics = multi_metrics, control = ctrl_oof)

# Collect metrics
rf_metrics  <- collect_metrics(rf_oof)  |> mutate(model = "ranger")
xgb_metrics <- collect_metrics(xgb_oof) |> mutate(model = "xgboost")

oof_metrics <- bind_rows(rf_metrics, xgb_metrics) |> arrange(.metric, desc(mean))

# Confusion matrices
rf_preds  <- collect_predictions(rf_oof)
xgb_preds <- collect_predictions(xgb_oof)

rf_cm  <- conf_mat(rf_preds,  truth = lithostrat_id, estimate = .pred_class)
xgb_cm <- conf_mat(xgb_preds, truth = lithostrat_id, estimate = .pred_class)
```

``` r
oof_metrics
```

    # A tibble: 6 × 7
      .metric      .estimator  mean     n std_err .config         model  
      <chr>        <chr>      <dbl> <int>   <dbl> <chr>           <chr>  
    1 accuracy     multiclass 0.602     5 0.0192  pre0_mod0_post0 ranger 
    2 accuracy     multiclass 0.600     5 0.0229  pre0_mod0_post0 xgboost
    3 bal_accuracy macro      0.707     2 0.00104 pre0_mod0_post0 ranger 
    4 bal_accuracy macro      0.702     2 0.0116  pre0_mod0_post0 xgboost
    5 kap          multiclass 0.543     5 0.0253  pre0_mod0_post0 xgboost
    6 kap          multiclass 0.542     5 0.0222  pre0_mod0_post0 ranger 

``` r
rf_cm 
```

                          Truth
    Prediction             Aalbeke Asse Bolderberg Brussel Diest Kwatrecht Lede
      Aalbeke                   21    4          0       0     0         3    0
      Asse                       2   19          0       0     0         3    1
      Bolderberg                 0    0          0       0     0         0    0
      Brussel                    0    0          0      16     2         0    9
      Diest                      0    0          0       0     0         0    0
      Kwatrecht                  2    2          0       0     0        10    0
      Lede                       0    0          3      13     1         0   42
      Maldegem                   0    0          0       0     0         0    0
      Merelbeke                  2    0          0       0     0         2    0
      Mons_en_Pevele             2    0          0       8     0         0    5
      Mont_Panisel               4    2          0       0     0         3    0
      Onbekend                   0    0          0       0     1         0    2
      Quartair                   3    0          4       6     2         0    1
      Sint_Huibrechts_Hern       0    0          0       0     0         0    0
      Ursel                      2    2          0       0     0         2    0
      Wemmel                     3    0          0       2     0         1    3
                          Truth
    Prediction             Maldegem Merelbeke Mons_en_Pevele Mont_Panisel Onbekend
      Aalbeke                     0         3              0            4        0
      Asse                        0         1              1            3        0
      Bolderberg                  0         0              0            0        0
      Brussel                     3         0              2            2        1
      Diest                       0         0              0            0        0
      Kwatrecht                   0         1              0            2        0
      Lede                        0         0              6            3        1
      Maldegem                    0         0              0            0        0
      Merelbeke                   0         1              0            0        0
      Mons_en_Pevele              1         0             32            3        0
      Mont_Panisel                0         1              0           27        0
      Onbekend                    0         0              0            0        9
      Quartair                    0         0              0            1       31
      Sint_Huibrechts_Hern        1         0              0            0        0
      Ursel                       0         3              0            1        0
      Wemmel                      1         0              5            5        0
                          Truth
    Prediction             Quartair Sint_Huibrechts_Hern Ursel Wemmel
      Aalbeke                     1                    0     1      1
      Asse                        0                    0     1      2
      Bolderberg                  0                    0     0      0
      Brussel                     0                    0     0      4
      Diest                       0                    0     0      0
      Kwatrecht                   0                    0     1      0
      Lede                        0                    0     0      6
      Maldegem                    0                    0     0      1
      Merelbeke                   0                    0     1      0
      Mons_en_Pevele              0                    0     0      1
      Mont_Panisel                0                    0     0      1
      Onbekend                    7                    0     0      0
      Quartair                  135                    4     2      0
      Sint_Huibrechts_Hern        0                    0     0      1
      Ursel                       0                    0    15      0
      Wemmel                      0                    7     0     28

``` r
xgb_cm
```

                          Truth
    Prediction             Aalbeke Asse Bolderberg Brussel Diest Kwatrecht Lede
      Aalbeke                   25    3          0       0     0         5    0
      Asse                       2   17          1       0     0         2    0
      Bolderberg                 0    0          0       0     0         0    1
      Brussel                    0    0          0      21     1         0   11
      Diest                      0    0          0       0     0         0    0
      Kwatrecht                  1    4          0       0     0         8    0
      Lede                       0    2          1      12     1         0   37
      Maldegem                   0    0          0       0     1         0    0
      Merelbeke                  1    0          0       0     0         2    0
      Mons_en_Pevele             2    0          0       5     0         0    7
      Mont_Panisel               6    0          0       0     0         2    1
      Onbekend                   0    0          0       0     1         0    1
      Quartair                   2    0          3       4     2         0    1
      Sint_Huibrechts_Hern       0    0          1       0     0         0    1
      Ursel                      2    3          1       0     0         4    0
      Wemmel                     0    0          0       3     0         1    3
                          Truth
    Prediction             Maldegem Merelbeke Mons_en_Pevele Mont_Panisel Onbekend
      Aalbeke                     0         3              0            2        0
      Asse                        0         1              0            1        0
      Bolderberg                  0         0              0            0        0
      Brussel                     0         0              3            0        0
      Diest                       0         0              0            0        0
      Kwatrecht                   0         0              1            1        0
      Lede                        3         0              5            3        0
      Maldegem                    0         0              0            0        0
      Merelbeke                   0         0              0            0        1
      Mons_en_Pevele              1         0             31            3        0
      Mont_Panisel                1         1              2           33        0
      Onbekend                    0         0              0            0       12
      Quartair                    0         0              0            1       29
      Sint_Huibrechts_Hern        0         0              0            0        0
      Ursel                       0         5              0            1        0
      Wemmel                      1         0              4            6        0
                          Truth
    Prediction             Quartair Sint_Huibrechts_Hern Ursel Wemmel
      Aalbeke                     0                    0     1      1
      Asse                        0                    1     1      2
      Bolderberg                  0                    1     0      0
      Brussel                     0                    0     0      2
      Diest                       0                    0     0      0
      Kwatrecht                   0                    0     3      0
      Lede                        1                    2     0      4
      Maldegem                    0                    0     0      0
      Merelbeke                   0                    0     0      0
      Mons_en_Pevele              0                    0     0      4
      Mont_Panisel                1                    1     0      2
      Onbekend                   13                    0     0      0
      Quartair                  127                    1     1      0
      Sint_Huibrechts_Hern        1                    0     0      2
      Ursel                       0                    0    15      0
      Wemmel                      0                    5     0     28

``` r
#
oof_wide <- oof_metrics |>
  dplyr::select(model, .metric, mean, n, std_err) |>
  tidyr::pivot_wider(names_from = model, values_from = c(mean, std_err, n))

kable(oof_wide)
```

| .metric | mean_ranger | mean_xgboost | std_err_ranger | std_err_xgboost | n_ranger | n_xgboost |
|:---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.6016565 | 0.6002609 | 0.0192155 | 0.0229184 | 5 | 5 |
| bal_accuracy | 0.7073291 | 0.7020837 | 0.0010412 | 0.0116062 | 2 | 2 |
| kap | 0.5421334 | 0.5430688 | 0.0222207 | 0.0253415 | 5 | 5 |
