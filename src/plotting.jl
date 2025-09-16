using LaTeXStrings
using CairoMakie

set_theme!(theme_latexfonts())

# Golden ratio
golden = (1 + sqrt(5)) / 2

width = 4.0
size_inches = (width, width / golden)
size_pt = 72 .* size_inches

dpi = 300

ax_kwargs = (
    titlealign = :left,
    xgridvisible = false,
    ygridvisible = false,
    spinewidth = 0.7,
    xtickwidth = 0.7,
    ytickwidth = 0.7
)

colors = Makie.wong_colors();

dpi = 300
inch = 96 # this is relative to 1 CSS px
pt = 72
fontsize_pt = 10
fontsize = 10 * inch / pt
