# Official cashbox visualization asset

`official_cashbox.glb` is a visualization-only derivative of the official
`cash_lux` cooked assets already shipped inside this ROBOTAC package. It contains
the exact 684-triangle mesh and the official green/gold base-color texture.
The local simulator only places this copy in RViz; it does not move, spawn,
hide, or otherwise modify any Unreal actor.

Source files and SHA-256:

- `StaticMeshes/cash_lux.uasset`: `bd76f765760fd9cb8bcf0d647db600d6a3cc0c439d48ed7c7a9618451fdefadf`
- `StaticMeshes/cash_lux.uexp`: `ca36c90578365872cbb0e371785d02a4ca1354781c0146116bf2eb233dfa7333`
- `StaticMeshes/cash_lux.ubulk`: `5185f17627559b271b77637fb9627df9d4c2a0be54691dc294649afec18e8b1b`
- `Textures/cash_lux_texture_0.uasset`: `ad92a9ec440dc180506b73f2004a96eabcebf45f576f913f9ea9540fb39f2127`
- `Textures/cash_lux_texture_0.uexp`: `8b6029240c21986745cc2528986c222ade4ac83bfb98f7cb4fb99ec2bda91585`
- `Textures/cash_lux_texture_0.ubulk`: `3ce48caa9b94c21594a4a303afb6920021ffeb569f4464c12697f14b9309b17c`
- Derived `official_cashbox.glb`: `b446b6ce2af715bb41dbb44bc76a7cf40bef94136383a55aadf5f32a745227c8`

The GLB is self-contained and has no network or external texture dependency.
It remains local proxy visualization and is not official score evidence.
