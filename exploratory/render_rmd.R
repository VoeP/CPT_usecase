
library(rmarkdown)
library(here)


render(here("modeling", "R_modelling.rmd"),
       output_format = "github_document",
       quiet = TRUE,
       clean = FALSE)
