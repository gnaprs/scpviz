All notable changes to this project are documented here.

<details open>
<summary><b>1.2.0</b> – 2026-11-08</summary>


#### Added


##### (Enrichment)


- Cache STRING IDs manually and enrich all DE genes by default (0885fde…)

- Add direction="both" and log STRING enrichment to history (56a209b…)



##### (PAnnData)


- Add .pdata save/load and export_h5ad (00df863…)

- Add .pdata save/load, STRING enrichment overhaul, and DE dict-repr keys (breaking) (8ac2590…)



##### (Utils)


- Add verbose progress logging to get_upset_query (982c7e2…)

- Support length in peptide property helpers (0f09cbf…)

- Store DE comparison labels as dict-repr in stats keys (39d96f2…)



#### Build System


##### (Docs)


- Upgrade mkdocstrings to 1.x and mkdocstrings-python to 2.x (b33ca6e…)



##### (V1.2.0)


- Bump version (774c3b0…)



#### CI


##### (Changelog)


- Classify breaking commits and skip changelog bot commits (70100a9…)



#### Changed


##### (Plotting)


- Merge heatmaps into clustering.py and polish plot_clustermap (dd5fd68…)



#### Chores


##### (Dev)


- Add save/load verification notebook (ed58611…)



##### (PAnnData)


- Remove unused tkinter import and dead summary copy (06404e6…)



##### (Todo)


- Mark rs sync bug as fixed, add bug on empty load() (eb22a1a…)



#### Documentation


##### (Enrichment)


- Refresh quickstart outputs and fix enrichment tutorial examples (9719f52…)

- Refresh single_cell outputs and regenerate enrichment figure (1301352…)




- Correct stale contributor and tutorial instructions (9004559…)



#### Fixed


##### (Enrichment)


- POST identifiers in plot_enrichment_svg to avoid STRING 520 on long lists (997d2c9…)



##### (Filtering)


- Sync peptides and rs correctly in in-place protein filters (c3d7bf0…)

- Align found/significance metrics to current features (94a0944…)



##### (PAnnData)


- Skip anndata 0.13 layers[None] in filter and repr (c8c36d7…)

- Build summary when constructing from prot/pep (a6120ef…)

- Refresh summary after leiden and pass harmony kwargs (312b89c…)



##### (Plotting)


- Use hollow circle edge legends and grey shape handles in PCA/UMAP (8a9f5a4…)

- Skip housekeeping groups with no matching genes (8b6bd4c…)




- Honor debug flag in plot_raincloud and get_upset_contents (f8a67c2…)



##### (Utils)


- Query UniProt via /accessions and keep entry-name stream fallback (e1b6445…)

- Raise clear KeyError for missing classes column in get_samplenames (f61c904…)

- Keep numeric peptide property columns as float (9452e8d…)



#### Other



</details>

<details open>
<summary><b>1.1.2</b> – 2026-08-05</summary>


#### Added


##### (PAnnData)


- Add donor-blocked mixed_de with volcano and docs (ebdc7b3…)

- Add donor-blocked mixed_de (0d36377…)



##### (Plotting)


- Add pc_pad and cbar_scale to PCA-GSEA bubble layout (e03f6ad…)

- Add grouped and clustered publication heatmaps (278e235…)

- Auto-size heatmap legend_width and document heatmaps (a33ebd0…)

- For heatmap plots add flexible handling for header_colors and expose dendrogram line width (7bbb13c…)

- Unify heatmap row/column spacing and document scale examples (86fdc18…)

- Expose header_height and group_bar_width for grouped and clustered heatmaps (42c3d8b…)

- Add grouped and clustered publication heatmaps (27bd2ae…)



##### (Utils)


- Add peptide ProtParam properties via BiophysicalMixin (9add2be…)



#### Build System


##### (V1.1.2)


- Bump version to 1.1.2 (00853ac…)



#### Documentation


##### (PAnnData)


- Add biophysical API page and fix hidden/PlotMixin refs (2c37304…)



##### (Plotting)


- Wire mapping and PCA-GSEA examples to generated figures (de32e86…)



#### Fixed


##### (PAnnData)


- Treat zero as missing in quant metrics and refresh summary on X changes #48 (9e71e21…)

- Fix _resolve_de_key failing on legacy dict key, fix cell-level LMM failing to converge and default to auto to fall back on pseudobulk (e067645…)

- Bug fix for mixed_de having mixedlm non-convergence on near-zero (sensitive to BLAS/OS ubuntu py3.11) (31b5d84…)



##### (Plotting)


- Pack overlapping significance brackets in abundance boxgrid (7e06ba8…)

- Scale PCA-GSEA bubbles to axes and clear size legend (6b4edb6…)

- Accept scalar keys for single-column mapping (501a389…)



#### Other


#### Style


##### (Docs)


- Fix formatting errors on docstrings (d79ef4f…)



##### (Filtering)


- Fix typo in docstring (889f32e…)



#### Tests


##### (PAnnData)


- Add PD and DIA-NN real-data smoke coverage for recent APIs (dae6e89…)



##### (Plotting)


- Update tests for grouped and clustered publication heatmaps (f33fbe9…)




</details>

<details open>
<summary><b>1.1.1</b> – 2026-07-09</summary>


#### Build System


##### (V1.1.1)


- Bump version to 1.1.1 (250ab3f…)



#### Fixed


##### (PAnnData)


- Treat zero as missing in quant metrics and refresh summary on X changes #48 (d6ad025…)



#### Style


##### (Docs)


- Fix formatting errors on docstrings (711842e…)




</details>

<details open>
<summary><b>1.1.0</b> – 2026-06-22</summary>


#### Added


##### (Plotting)


- Add significance brackets to plot_abundance_boxgrid (cb66595…)

- Add abundance colorbar_norm and nan_color to PCA/UMAP #43 #47 (c671c8b…)

- Add significance brackets to plot_abundance_boxgrid, add abundance colorbar_norm and nan_color to PCA/UMAP (ca2453e…)



#### Build System


##### (V1.1.0)


- Bump version to 1.1.0 (d9179a3…)



#### Documentation


##### (PAnnData)


- Expand plot mixin docstrings to match plotting module (04107e6…)



##### (Tutorials)


- Add boxgrid significance figures to quickstart and single_cell (a826945…)

- Add boxgrid significance figures to quickstart and single_cell (278bf6b…)



#### Fixed


##### (Plotting)


- Facet error on py3.11 (5e301e3…)

- Apply log y-scale before facet violin strip overlay (f8f4c2c…)

- Harden faceted violin strip overlay on headless backends (eca33bc…)



#### Other


##### (Paper)


- Update Paper PDF Draft (05bed69…)

- Update Paper PDF Draft (ca69df0…)



#### Tests


##### (Plotting)


- Cover boxgrid significance and facet branches in abundance.py (8ab3807…)




</details>

<details open>
<summary><b>1.0.1</b> – 2026-06-15</summary>


#### Added


##### (Analysis)


- Add BH FDR correction to DE/volcano plots (c443206…)

- Add pval deprecation warning instead of breaking change from BH FDR correction commit (e3da0fe…)



##### (Filtering)


- Add filter_sample POI mode and refactor filter_prot #46 (4be9906…)



##### (Import)


- Update import to bypass API calls with fetch_uniprot (db9fd94…)



##### (Plotting)


- Enhance plot_cv with percent scale, annotations, and docs, closes #46 (21e4ca9…)



##### (Utils)


- Update get_upset_query to bypass API fetch with fetch_uniprot (76f6e99…)

- Add RS-based accession–peptide lookup helpers (459c922…)

- Extend resolve_accessions with all_matches and add resolve_peptides (5cae58d…)



#### Build System


##### (Dependencies)


- Update scipy min version to 1.11 for BH FDR implementation (20c7a82…)



##### (Joss)


- Update citation links to JOSS (90a1596…)



##### (V1.0.1)


- Bump version to 1.0.1 (47e8628…)



#### Documentation


##### (Tutorials)


- Document POI filtering and all_matches in filtering tutorial #46 (d504013…)



#### Fixed


##### (Plotting)


- Use matplotlib colormap registry and harden facet plots to handle update to matplotlib 3.11 (9995a2e…)

- Set log scale before facet legend in plot_abundance (982bff0…)



#### Other


##### (Paper)


- Update paper wrt editor comments (00d7f13…)

- Update affiliations (d3a98d0…)

- Update Paper PDF Draft (8fee1a7…)

- Update Paper PDF Draft (9d26b1c…)



#### Style


##### (Utils)


- Fix convert_identifiers docstring formatting and add examples (bc57970…)




</details>

<details open>
<summary><b>1.0.0</b> – 2026-05-21</summary>


#### Build System


##### (V1.0.0)


- Bump version to 1.0.0 (c3e1193…)



#### Documentation


##### (Plotting)


- Update plotting API with figure example images close #45 (7a0fcae…)



#### Fixed


##### (Build)


- Set pyharmony requirement <1.0 to avoid C++ requirement on CI windows build fix #44 (5de43f9…)

- Force fixed pyharmony version (bc76b8e…)



#### Other


##### (Paper)


- Update manuscript for reviewer feedback (6967ffe…)

- Update Paper PDF Draft (32f6b99…)

- Update manuscript (1f5939e…)

- Update Paper PDF Draft (250ac22…)



##### (Plotting)


- Add mapping key for easier plotting, fix legends overlap and shift_legend not working on multiple legend, fix #42 (20df3ae…)




</details>

<details open>
<summary><b>0.6.0</b> – 2026-04-14</summary>


#### Added


##### (Analysis)


- Add ssGSEA and declare gseapy dependency (33a3578…)

- Add pairwise_correlation function (d5a62b9…)

- Add log_transform for preprocessing and update handling of log data on de(), pca() and pairwise_correlation() (e8f9af4…)



##### (Editing)


- Update export_layer to support multiple labels (25dd299…)



##### (Enrichment)


- Add enrichment_functional_pca for STRING on PCA loadings (0099128…)



##### (PAnnData)


- Layer provenance registry, log_transform, and show_layer_provenance (5cceb9a…)

- Update layer provenance in .analysis functions (c1cca58…)



##### (Pca_gsea)


- Implement gseapy prerank, using multi-library default, FDR counts in verbose (1b6cd7d…)



##### (Plotting)


- Add PCA-GSEA plots and pAnnData wrappers (44d82f2…)

- Add plot_pairwise_correlation (86483c3…)

- Add accession and gene columns to plot_cv() returned df (ac850b9…)



#### Build System


##### (Deps)


- Restructure dependencies with optional extras and bump min Python to 3.10 #33 (12f5dbd…)

- Move openpyxl to core and set min versions for [dev] #30 #31 (0024f0d…)



##### (Setup)


- Update setup.py with updated deps (87fe954…)



##### (V0.6.0)


- Update to stable launch, add gseapy analysis and plotting, refactor plotting and utils (d3fbfa7…)



#### CI


##### (Docs)


- Update to local editable copy of scpviz for docs serve (f069efa…)




- Automate changelog and PyPI release workflows (a773a00…)



#### Changed


##### (Plotting)


- Cleanup, move single use functions into helper functions (643061c…)

- Refactor both biplots for protein and pathway vectors to accept n_vectors per pc, fix bug on ylim/xlim, allow cmap coloring (3112be7…)

- Plotting module refactor (3da4e61…)



##### (Utils)


- Utils module refactor (b54988f…)



#### Chores


##### (Gitignore)


- Extend .gitignore for save tar folders etc (a89b398…)



##### (Merge)


- Merge main into development, resolve conflicts (80c9dc7…)

- Merge changes to .gitignore (491ee9a…)



##### (Todo)


- Refactor todo list w priority and better oganization (cf4d03c…)




- Expose __version__, align print_versions, ignore logo asset (e750e24…)



#### Documentation


##### (Links)


- Fix wrong/dead links (5d28169…)



##### (Utils)


- Synchronize docs layout for utilies and plotting modules (d83cb71…)



##### (Version bump)


- Document optional sc extras and update Python requirement to 3.10 #31 #33 (85d10c8…)



#### Fixed


##### (Plotting)


- Update plot_venn to override alpha if provided (32afcbe…)



##### (Test_provenance_chain_depth_after_chain)


- Add update to current_X_layer in editing.py (ae7d7de…)



##### (Tests)


- Correct GLOBAL_DEBUG import in pytest_configure (4d7a271…)

- Typo in test assertion for test_print_versions_runs() (ffb3b0a…)



##### (Utils)


- Force string typecast on join for get_classlist() (474f97b…)



#### Other


##### (Paper)


- Update Paper PDF Draft (ebe6943…)

- Update Paper PDF Draft (f4b1e86…)

- Update Paper PDF Draft (6451bae…)



#### Style


##### (Analysis)


- Fix typo in pca_gsea() printed results key (f28f201…)



##### (Docs)


- Fix griffe errors for plot.py (c3cf62d…)

- Fix griffe errors for pAnnData.py (f7dd584…)



#### Tests


##### (Analysis)


- Cover ssGSEA storage, gene column, and duplicate collapse (e1e8d04…)

- Remove xfail from passing tests due to bump in python version (634addd…)



##### (Editing)


- Add tests for layer provenance (2b13eac…)



##### (Enrichment)


- Cover enrichment_functional_pca PC tails and key suffix (8ca4855…)



##### (Import)


- Debug test_import_pd_excel #30 (cd4c98c…)

- Exclude macos py3.8 due to scikit installation error (7efb218…)



##### (Plotting)


- Cover PCA-GSEA plots, protein loading vectors, and pAnnData wrappers (297e698…)




</details>

<details open>
<summary><b>0.5.9</b> – 2026-04-08</summary>


#### Build System


##### (V0.5.9)


- Bump version to 0.5.9 (8599fc8…)



#### Chores


##### (Todo)


- Refactor todo list w priority and better oganization (e6d3c4b…)



#### Documentation


##### (Tutorial)


- Update single cell tutorial to push deployment again (285510f…)




- Docs(citation): add CITATION.cff, closes [JOSS] Add CITATION.cff
Fixes #32 (f1d7bfe…)

- Docs(makefile): add makefile, closes [JOSS] No Makefile/Justfile
Fixes #35 (198692f…)



#### Fixed


##### (Docs)


- Update deployment workflow to run only on push to docs on main branch (830bbb3…)

- Fixed broken link to function hyperlink, added colab button (7662ea8…)



#### Other


##### (Paper)


- Update Paper PDF Draft (da1b895…)

- Update Paper PDF Draft (01dbb0a…)

- Update research impact statement regarding collaborative effort #36 (b3e540a…)

- Update Paper PDF Draft (3d2700f…)

- Update bib to remove local file paths #28 (1b11b77…)

- Update Paper PDF Draft (148081f…)



#### Style


##### (Docs)


- Fix formatting in single-cell tutorial header (73970d3…)



##### (Readme)


- Update contributing and license (cbb355b…)




</details>

<details open>
<summary><b>0.5.8-alpha</b> – 2026-03-16</summary>


#### Added


##### (Plotting)


- Update plot_abundance_boxgrid to more flexible plotting style, bug fixes (4276636…)

- Update plot_abundance_boxgrid and add tests (2c1b0cd…)



#### Build System


##### (V0.5.8a)


- Update to new patch with additional tutorials (42ac903…)



#### Documentation


##### (Plotting)


- Update examples for various plotting functions (cc43b62…)



##### (Quickstart)


- Update quickstart to include new plotting functions (d4f1d48…)



##### (Tutorial)


- Add wget downloads to importing ipynb (f9d34fa…)

- Update single cell and quickstart tutorial (1a04522…)

- Update single_cell.ipynb (b5eb1fb…)



##### (Tutorials)


- Complete importing tutorial (3534a19…)



##### (Upload)


- Upload test_input.csv (94ca178…)



#### Fixed


##### (Get_abundance)


- Bug fix on get_abundance throwing error when passing in string instead of list (9b70b75…)



##### (Tests)


- Typos on errortype and assertion match (f0b9674…)



#### Other


#### Tests


##### (Import)


- Add test for pep xlsx import (6d67ccb…)



##### (Plotting)


- Update test names for plot_abundance_boxgrid (21933a5…)




</details>

<details open>
<summary><b>0.5.7-alpha</b> – 2026-01-29</summary>


#### Added


##### (Analysis)


- Add pimms imputation (3745120…)



##### (Directlfq)


- Make strict/loose flag instead to replicate older behavior (d976225…)



##### (Enrichment)


- Allow category argument for plot_enrichment_svg (297e9f7…)



##### (Filtering)


- Add warning about underscores in class_type throwing error within annotate_found and annotate_significant (51b0664…)



##### (Plot_abundance_boxgrid)


- Add publication quality abundance plot function (2957b40…)



##### (Plotting)


- Add classes=None support to plot_abundance_boxgrid, also fixed flipped label error by enforcing order from start (27043ea…)

- Add shift_legend() convenience function (53a95df…)

- Add ax input for plot_abundance_boxgrid, update wrapper in pAnnData (2caf525…)

- Add plot_volcano_adata to support transcriptomics plotting, add mark_volcano_by_significance for more flexibility with marks (e859a97…)

- Add flexible plotting options to plot_pca and plot_umap (dc36d29…)

- Add text adjustments to plot_volcano (4580a12…)

- Text adjustments to mark_volcano and mark_volcano_significance (2c8219a…)

- Add weighted option to plot_venn (4e516a4…)



##### (Utils)


- Add parse_filename_index to help with formatting imports (eb9da97…)

- Add de_adata to support transcriptomics DE (e0cdb74…)

- Add condition flag for parse_filename_index to parse subset of samples, added tests for condition (09ce308…)

- Additional fixes to de_adata() (0a09fda…)



#### Build System


##### (.gitignore)


- Add parquet file (eecb64d…)



##### (Docs)


- Enforce version type on mkdocstrings since new build broke render (36b3734…)

- Fix versions of mkdocstrings (feec11a…)



##### (V0.5.3a)


- Update readme and dependencies (4abf8c2…)



##### (V0.5.6a)


- Update dependencies and bump to updated readme link (4239e51…)



##### (V0.5.7a)


- Update to new patch with plotting overhaul (9bfed9e…)

- Update to new patch with plotting overhaul (c0a92bf…)



#### CI


##### (Codecov)


- Update patch pass to 80% (809e06a…)

- Patch pass to 70% (c19c842…)



##### (Package)


- Modify readme to show status matrix (5202a0e…)

- Split ci tests into individual workflows (59dd2c9…)

- Update with codecov token (24f61bf…)

- Fix secrets passthrough to reusable workflow (70fd891…)



##### (Pytest)


- Update to test on macos and windows as well (66afe02…)



#### Changed


##### (Directlfq)


- Default to strict (876a237…)



##### (Enrichment)


- Moved get_string_mappings() to utils to for ease of use for user, refactored enrichment mixin (b8cfc76…)



##### (Git LFS)


- Remove large tutorial parquet from LFS; keep small test parquet (5c46f16…)

- Remove LFS flag from pytest workflow (b8b98c4…)

- Move large parquet file to release and update download links in docs (dafcc43…)

- Move small test parquet from LFS to local (c0170bf…)



##### (Joss)


- Update to new joss paper format (46e5048…)

- Added state of field section (18d4a4b…)



##### (Plotting)


- Add _add_continuous_colorbar helper function, clean plot_pca and plot_umap code (60c724a…)

- Update pca and umap to pass kwargs to scatter plot (fffa5e1…)

- Enable label_x for box=True in plot_abundance_boxgrid (d789d43…)

- Refactor volcano adata (0b2079d…)

- Update docstrings, update references to old uniprot naming (64af394…)

- Reorganize imports (56d478b…)

- Overhaul plot_umap and plot_pca to use common _plot_embedding_scatter, revamp ellipse handling (026ef71…)

- Return ax on plot_venn to match other plot functions, change test (5722d3c…)



##### (Setup)


- Add GLOBAL_DEBUG flag to suppress runtime warnings for user but allow at pytest logging (ad48d06…)



#### Chores


##### (Codecov)


- Set success flag for ci (0f1f2bc…)



##### (Plot_abundance)


- Default to raw abundance with y-log scale (92beb20…)



##### (Readme)


- Update CI badge link in README.md (8ef019a…)



#### Documentation


##### (Refactor)


- Overhaul documentation layout for API pAnnData reference (df24ead…)




- General updates to site style, formatting for mixins (d0a1bca…)



#### Fixed


##### (Analysis)


- Suppress warnings about nanmean (3b2f505…)



##### (Directlfq)


- Handle bug with multi proteins associated with same peptide in output file producing NaNs output (e.g. 'P03995;P03995-2') (ef8a915…)

- Move expansion to peptide level data so normalization algorithm maps to correct protein (f676b44…)



##### (Docs)


- Fix link to importing tutorial (541b849…)



##### (Io)


- Bug fix on _import_proteomeDiscoverer() where peptide has no matching protein (248e03d…)



##### (Mkdocstrings)


- Downgrade to v0.30.1 for now while we figure out handler error (7d7ba8c…)



##### (Plot_umap)


- Fix force to propagate through umap, neighbors and pca (9092ebb…)



##### (Plotting)


- Bug fix for plot_abundance_boxgrid when xtick only exists for one class (4e267fd…)

- Default plot_abundnace_boxgrid to dodge=False when classes is None (6290754…)

- Fix GLOBAL_DEBUG flag to be updated at start of test_utils.py (4c84318…)

- Allow n_comps or n_components (as per error message) to be passed to pca_params in plot_pca (0005636…)

- Fix return on plot_venn to pass test (6d1709c…)



##### (Summary)


- Check and remove_unused_categories after filter, mostly for PD data (925e72d…)



##### (Umap)


- Fix force=True not propagating down to neighbors() and pca() (d75f1ed…)



##### (Utils)


- Add warning/hint about source of length mismatch error in format_class_filter (664bfdf…)

- Fix for py3.8 compatibility (8163c5c…)



#### Other


##### (Paper)


- Update Paper PDF Draft (4644379…)

- Update Paper PDF Draft (c371cd7…)

- Update Paper PDF Draft (d65ce10…)



#### Style


##### (Analysis)


- Remove extra blank lines (1897136…)



##### (Docs)


- Change root_heading to true, remove others section (bb68f1d…)

- Add tabs back to website for better navigation (45ba11f…)



##### (Editing)


- Edit mixin docstring to include export_layer (b3d2cba…)



##### (Enrichment)


- Add categories to docstring of plot_enrichment_svg (06858b2…)

- Add catch for no enrichment result (85c0399…)



##### (Filter_prot_found)


- Add new line after rs message (7a4146b…)

- Reorder automatic annotation message (9147fea…)



##### (Filtering)


- Refactor the print statements from filter_prot_found to match filter_prot_significant, fix style for filter_prot valid genes and unique profile (1660e0b…)



##### (Plotting)


- More verbose error message for errors in resolve_plot_colors (1f2dc63…)



##### (Readme)


- Update badges (f773576…)



##### (Rs)


- Show rs filter message after instead of before printouts (2d1803f…)



##### (Utils)


- Fix formatting on example for get_string_mappings (7f38f7c…)



#### Tests


##### (Add files)


- Add files for tests (5718f9e…)



##### (Analysis)


- Add mark xfail for failure due to potential package version mismatch (a7c808a…)



##### (Boxgrid)


- Fix boxgrid assertion error (c8c204f…)



##### (De_adata)


- Refactor raise error to earlier in check for method (41ba612…)



##### (Directlfq)


- Add tests for strict flag and no pep situation (cd05079…)



##### (Enrichment)


- Add test for helper _scalarize_taxon() within get_string_mappings (485b830…)



##### (Filter_prot_found)


- Add test for automated annotation with simple group (451ce0a…)



##### (Impute)


- Add test for use_zeros_as_nan flag (32fa2bd…)



##### (PAnnData_plot)


- Refactor base plot tests into test_pAnnData_plot file, centralize where to test pdata plotting functions (3c4c264…)



##### (Pimms)


- Add tests for pimm imputation (d3a1832…)

- Add skip for pimms-learn dependency issue (6485e2e…)



##### (Plotting)


- Add tests for plot_abundance_boxgrid (a42eb76…)

- Add test for abundance_boxgrid classes = None case (121a9f5…)

- Add test for abundance_boxgrid when show_n is true and parameterize box (e8aaf3b…)

- Add test for plot_umap with classes as None (6bbbe6c…)

- Add tests for shift_legend() (ed80cd9…)

- Add mark xfail for known plt version error, add test for violin (c6a3854…)

- Add test for plot_volcano_adata with no data (6791ad6…)



##### (Umap)


- Update larger perturbation on force_neighbors to pass test on py3.11 (more stable umap) (571149f…)

- Remove assertion on neighbour due to different stabilities across python versions (60badf5…)



##### (Utils)


- Add tests to exception and error for parse_filename_index (2446b41…)

- Add test for parse_filename_index (ae713de…)

- Add tests for get_string_mappings() (a31e3cd…)




</details>

<details open>
<summary><b>0.5.2-alpha</b> – 2025-11-16</summary>


#### Added


##### (Io)


- Add delimiter support for parsing imports (b1a23eb…)



#### Build System


##### (Dependencies)


- Add directlfq (c6dd2c3…)



#### CI


##### (Changelog)


- Add two minute sleep timer to prevent clash with joss workflow (478bacd…)

- Add rebase after sleep (4500b28…)



##### (Joss)


- Update workflow to pull before pushing due to changelog upload (514d580…)

- Remove dynamic ref logic to fix wrong head (ee792f4…)

- Add local checkout to main for pull (959a605…)



#### Fixed


##### (Ci)


- Parquets were updated to lfs, add lfs request to pytest checks (391bb8b…)

- Add lfs update to python-package.yml (13f335f…)

- Trigger pytest workflow (1a44b79…)



##### (Lfs)


- Fix LFS tracking for DIANN parquet fixture (9417883…)



#### Other


##### (Paper)


- Update with comments from coauthors (c916be0…)

- Update Paper PDF Draft (a637997…)

- Update references (0b80c03…)

- Update Paper PDF Draft (74ab75a…)

- Final updates to paper.md (20b500c…)

- Update acknowledgements (0eb8e8b…)

- Update references (6cb919d…)

- Update Paper PDF Draft (4d41ca6…)

- Update Paper PDF Draft (7bf727d…)



#### Tests


##### (Io)


- Add python check to mapping function to account for function deprecation in pandas (8b6b130…)




</details>

<details open>
<summary><b>0.5.1-alpha</b> – 2025-11-10</summary>


#### Added


##### (Filtering)


- Add exclude_file_list argument (2f592f1…)



##### (Plot_cv)


- Add palette flag (f92100b…)



#### Build System


##### (Package name)


- Rename scviz to scpviz (9bc6347…)




- Bump scpviz to v0.5.0-alpha (1942e28…)



#### CI


##### (Changelog)


- Update .git-cliff.toml (85f2304…)



##### (Joss)


- Rename joss commit message to match conventional commit style (83c4935…)




- Update test CI to run only on changes to /src or /tests (06b3dc6…)



#### Chores


##### (Assets)


- Move assets out of src (917be4c…)



##### (Git lfs)


- Update .gitattributes for lfs upload of parquet files (ccf0e04…)



##### (Todo)


- Update todo list (388fdf3…)



#### Documentation


##### (Assets)


- Upload logo and test assets (e1ee49b…)



##### (Filtering)


- Finished filtering tutorial, update docstrings (9a3f6a5…)



##### (Package rename)


- Rename all links in docs (cfc8344…)

- Update mkdocs.yml (838fa33…)



##### (Quickstart)


- Update tutorial files (422ae4f…)

- Upload diann_report.parquet, using git lfs (32501ee…)

- Update download links (22c60d9…)



##### (Quicstart)


- Add colab link for user ease of use (8fbf44c…)



##### (Readme)


- Update readme (b8637fb…)

- Update broken links and logo (8d7b64a…)



##### (Setup)


- Update mkdocs.yml, add dev and js for navigation (6427731…)



##### (Tutorial)


- Update quickstart, some tutorials in works (91e10bf…)

- Update tutorial home page (f1a0cb9…)



##### (Tutorials)


- Add pending notice (23b7d0c…)




- Update readme to proper image (7b82bea…)

- Remove conda installation from readme, update gitignore (0e5e5ac…)

- Update joss paper.md (11151fc…)



#### Fixed


##### (Io)


- Implement handler for diann files when using suggest_obs_columns (93cfac5…)

- Push python version fix for .map() (49be220…)



#### Other


##### (Paper)


- Update Paper PDF Draft (2fe3401…)

- Update paper.md and bib (f587cb5…)

- Update Paper PDF Draft (8c63425…)



#### Performance


##### (Io)


- Initialize rs as sparse, memory improvement from 6+ GB usage to ~60MB (da721b6…)



#### Style


##### (Cv)


- Default to false verbose on cv resolve_class_filter (ee346ca…)



##### (Filtering)


- Fix print statements to be more verbose (2fdeae3…)

- Updated print statements to include exclude_file_list (3efd2c6…)

- Add print for cleanup with no empty prots (b71aab6…)

- Fix typo in print statement of annotate_significant_prot (9445f0a…)



##### (Readme)


- Update version on docs badge (ad74fa6…)




</details>

<details open>
<summary><b>0.4.1-alpha</b> – 2025-11-04</summary>


#### Added


##### (Base)


- Add compare_current_to_raw and get_X_raw_aligned functionality (043251b…)



##### (Filtering)


- Add valid_genes, unique_profiles to filter_prot and cleanup (nans) to filter_sample (e251c8a…)

- Add handling of duplicate gene name in filter_prot (01d750e…)



##### (Import)


- Add cleanup after import (b290ec8…)

- Add support for pd3.2 import (592a814…)



##### (Plot)


- Add plot_abundnace wrapper from pdata (8e2a01c…)



#### Build System



- Update pyproject.toml (18af439…)

- Update pyproject.toml and workflow bug (19f4c7a…)

- Fix changelog yml tag fetch error (5b20d69…)



#### CI


##### (Changelog)


- Upload changelog.md (2be787e…)



##### (Pytest.ini)


- Add test "slow" marker (8280133…)



#### Chores


##### (Changelog)


- Changelog sync to docs (84e991d…)

- Final updates (6b94c5b…)

- Edit changelog workflow (0ec6ab5…)



##### (Docs)


- Update deploy workflow to use committed changelog (9fa1705…)




- Fix github workflow bugs (1ef2b00…)

- Fix workflow yml (cc2544f…)

- Fix changelog yml workflow (cb956b1…)

- Fix changelog toml format (7c5dcc8…)

- Fix attempt for workflow (506109b…)

- Update full changelog [skip ci] (b8b5634…)



##### (Workflow)


- Build and deploy runs after changelog is finished (97491c3…)



#### Documentation


##### (Coc)


- Add contributor covenant code of conduct (72fd27e…)



##### (Contributing)


- Update contributing.md file (4002003…)




- Include changelog in docs (9d7dbc0…)

- Update markdown files (d72e6fa…)



#### Fixed


##### (Base)


- Anndata automatically aligns X_raw, removed function and tests (ce82a52…)



##### (Export)


- Handle export of .X (a3028bd…)



##### (Filtering)


- Fix bug on import with non-matching obs/summary after nan protein cleanup (f61c0b5…)



##### (Import)


- Renaming scheme for pd prot var (9bcbf44…)



##### (Plotting)


- Fix bug for volcano_df handling of "not comparable" (1d3d87b…)



#### Other


##### (Identifier)


- Fix indentation on update_identifiers tip (90b7bec…)



#### Style


##### (Changelog)


- Edit markdown formatting for docs changelog (ff92df7…)

- Update parsers to match conventional commit format (a870d32…)



##### (De)


- Add comment on metaboanalyst median normalization (42c0260…)



##### (Summary)


- Edit table of usage scenarios for update_summary to be clearer (ffd2a37…)



#### Tests


##### (Filtering,import)


- Add tests for duplicate gene handling, import pd32 (4cf1b86…)



##### (Test files)


- Add test for pd3.2 import with prot and pep, upload pd3.2 mock files (3c96c45…)




</details>

<details open>
<summary><b>0.4.0-alpha</b> – 2025-10-28</summary>


#### Added


#### Changed


#### Chores



- Add git-cliff config and changelog workflows (0159d05…)



#### Fixed


#### Other



</details>

<details open>
<summary><b>0.3.0-alpha</b> – 2025-10-08</summary>


#### Added


#### Changed


#### Documentation


#### Fixed


#### Other


#### Tests



</details>
<!-- generated by git-cliff -->
