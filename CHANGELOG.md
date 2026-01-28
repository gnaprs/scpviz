All notable changes to this project are documented here.

<details open>
<summary><b>Unreleased</b></summary>


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



##### (Setup)


- Add GLOBAL_DEBUG flag to suppress runtime warnings for user but allow at pytest logging (ad48d06…)



#### Chores


##### (Codecov)


- Set success flag for ci (0f1f2bc…)



##### (Plot_abundance)


- Default to raw abundance with y-log scale (92beb20…)



##### (Readme)


- Update CI badge link in README.md (8ef019a…)




- Update changelogs [skip ci] (9b4fecb…)

- Update changelogs [skip ci] (0a63f6b…)

- Update changelogs [skip ci] (525bdb9…)

- Update changelogs [skip ci] (6c89f44…)

- Update changelogs [skip ci] (fe0eb7b…)

- Update changelogs [skip ci] (48a0c89…)

- Update changelogs [skip ci] (2bc44af…)

- Update changelogs [skip ci] (a9ebeff…)



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



##### (Mkdocstrings)


- Downgrade to v0.30.1 for now while we figure out handler error (7d7ba8c…)



##### (Plot_umap)


- Fix force to propagate through umap, neighbors and pca (9092ebb…)



##### (Plotting)


- Bug fix for plot_abundance_boxgrid when xtick only exists for one class (4e267fd…)

- Default plot_abundnace_boxgrid to dodge=False when classes is None (6290754…)

- Fix GLOBAL_DEBUG flag to be updated at start of test_utils.py (4c84318…)

- Allow n_comps or n_components (as per error message) to be passed to pca_params in plot_pca (0005636…)



##### (Summary)


- Check and remove_unused_categories after filter, mostly for PD data (925e72d…)



##### (Umap)


- Fix force=True not propagating down to neighbors() and pca() (d75f1ed…)



##### (Utils)


- Add warning/hint about source of length mismatch error in format_class_filter (664bfdf…)



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



#### Tests


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



##### (Umap)


- Update larger perturbation on force_neighbors to pass test on py3.11 (more stable umap) (571149f…)

- Remove assertion on neighbour due to different stabilities across python versions (60badf5…)



##### (Utils)


- Add tests to exception and error for parse_filename_index (2446b41…)

- Add test for parse_filename_index (ae713de…)




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



#### Chores



- Update changelogs [skip ci] (20d06bc…)

- Update changelogs [skip ci] (9e5bac4…)

- Update changelogs [skip ci] (3e16c2e…)

- Update changelogs [skip ci] (752f655…)

- Update changelogs [skip ci] (3b72040…)

- Update changelogs [skip ci] (ddbbfd8…)

- Update changelogs [skip ci] (0033c88…)

- Update changelogs [skip ci] (5201ba9…)

- Update changelogs [skip ci] (366a05d…)

- Update changelogs [skip ci] (d35654a…)

- Update changelogs [skip ci] (1450a6d…)

- Update changelogs [skip ci] (14c452a…)

- Update changelogs [skip ci] (f920f5b…)

- Update changelogs [skip ci] (b73cbc1…)

- Update changelogs [skip ci] (2ee80b8…)

- Update changelogs [skip ci] (73f898a…)



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




- Update changelogs [skip ci] (8e09ce2…)

- Update changelogs [skip ci] (dfff921…)

- Update changelogs [skip ci] (6e3f746…)

- Update changelogs [skip ci] (8a99267…)

- Update changelogs [skip ci] (d6d3633…)

- Update changelogs [skip ci] (f0cbb17…)

- Update changelogs [skip ci] (00d4bf8…)

- Update changelogs [skip ci] (f849dd9…)

- Update changelogs [skip ci] (aa84d7c…)

- Update changelogs [skip ci] (11a3754…)

- Update changelogs [skip ci] (b7f7e72…)



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

- Update changelogs [skip ci] (c3a8704…)

- Update changelogs [skip ci] (2356fff…)

- Update changelogs [skip ci] (9910211…)

- Update changelogs [skip ci] (b5d04c7…)

- Update changelogs [skip ci] (7e5d8b6…)

- Update changelogs [skip ci] (a688e34…)

- Update changelogs [skip ci] (265982c…)

- Update changelogs [skip ci] (ca558ad…)



##### (Workflow)


- Build and deploy runs after changelog is finished (97491c3…)



#### Documentation


##### (Coc)


- Add contributor covenant code of conduct (72fd27e…)



##### (Contributing)


- Update contributing.md file (4002003…)




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



##### (Package name)


- Rename scviz to scpviz (9bc6347…)



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


#### Changed


#### Chores



- Add git-cliff config and changelog workflows (0159d05…)




</details>

<details open>
<summary><b>0.3.0-alpha</b> – 2025-10-08</summary>


#### Added


#### Changed


#### Documentation



- Include changelog in docs (9d7dbc0…)



#### Fixed


#### Other


#### Tests



</details>
<!-- generated by git-cliff -->
