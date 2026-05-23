All notable changes to this project are documented here.

<details open>
<summary><b>Unreleased</b></summary>


#### Added

##### (Dash App)

- TL;DR: The app is now deploying on render: https://scpviz-webapp.onrender.com/
- Note: It might take a few minutes for the app to load.

- Add full Dash workflow UI and callbacks for import, QC, preprocessing, embeddings, DE, STRING enrichment, and export bundle flow.
- Add in-app Plot Editor tab with SVG load/save/download flow and bridge assets for browser-side editing.
- Add context entry buttons to open editor directly from DE and enrichment tabs.
- Add DE volcano style controls (color/font/cutoffs), label manager, and label utilities:
  - add from selection/click
  - add by list or cutoff rules
  - exact vs substring token matching
  - label limit warning and max label cap
  - priority sort for rule-based auto-labeling
- Add native popup color-picker utilities in DE tab via frontend bridge.
- Add SOP + troubleshooting guide to README for end-to-end app operation.
- Other mix bug fixes, tolerance strengthening, outlier input catches, etc.

