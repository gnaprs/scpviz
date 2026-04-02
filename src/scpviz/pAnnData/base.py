import copy

from scpviz import utils
from scpviz.utils import format_log_prefix

class BaseMixin:
    """
    Core base methods for pAnnData.

    This mixin provides essential utility and management functions for cloning, 
    checking, and managing core attributes of a `pAnnData` object. These methods
    serve as foundational building blocks for other mixins and functions.

    Features:
    
    - Checks presence of data (.prot or .pep)
    - Safe object copying with state preservation
    - Internal metadata management (stats, history, summary)

    Functions:
        _has_data: Check whether .prot and/or .pep data are present  
        copy: Return a new `pAnnData` object with retained internal state
        show_layer_provenance: Print preprocessing provenance for one or all layers
    """
    def _has_data(self) -> bool:
        """
        Check whether the pAnnData object contains either protein or peptide data.

        Returns:
            bool: True if either .prot or .pep is not None; otherwise False.
        """
        return self.prot is not None or self.pep is not None # type: ignore[attr-defined]

    def copy(self):
        """
        Return a new `pAnnData` object with the current state of all components.

        This method performs a shallow copy of core data (.prot, .pep) and a deep copy of internal attributes
        (e.g., RS matrix, summary, stats, and cached maps). It avoids full deepcopy for efficiency and retains
        the current filtered or processed state of the object.

        Returns:
            pAnnData: A new object containing copies of the current data and metadata.
        """
        new_obj = self.__class__.__new__(self.__class__)

        # Copy core AnnData components
        new_obj.prot = self.prot.copy() if self.prot is not None else None # type: ignore[attr-defined]
        new_obj.pep = self.pep.copy() if self.pep is not None else None # type: ignore[attr-defined]
        new_obj._rs = copy.deepcopy(self._rs) # type: ignore[attr-defined]

        # Copy summary and stats
        new_obj._stats = copy.deepcopy(self._stats) # type: ignore[attr-defined]
        new_obj._history = copy.deepcopy(self._history) # type: ignore[attr-defined]
        new_obj._suppress_summary_log = True # type: ignore[attr-defined]
        new_obj.summary = self._summary.copy(deep=True) if self._summary is not None else None # go through setter to mark as stale, # type: ignore[attr-defined]
        del new_obj._suppress_summary_log # type: ignore[attr-defined]

        # Optional: cached maps
        if hasattr(self, "_gene_maps_protein"):
            new_obj._gene_maps_protein = copy.deepcopy(self._gene_maps_protein) # type: ignore[attr-defined]
        if hasattr(self, "_protein_maps_peptide"):
            new_obj._protein_maps_peptide = copy.deepcopy(self._protein_maps_peptide) # type: ignore[attr-defined]

        return new_obj

    def compare_current_to_raw(self, on="protein"):
        """
        Compare current pdata object to original raw data, showing how many samples and features were dropped.
        Compares current obs/var names to the original raw data (stored in .uns).

        Args:
            on (str): Dataset to compare ('protein' or 'peptide').

        Returns:
            dict: Dictionary summarizing dropped samples and features.
        """
        print(f"{format_log_prefix('user', 1)} Comparing current pdata to X_raw [{on}]:")

        adata = getattr(self, "prot" if on == "protein" else "pep", None)
        if adata is None:
            print(f"{format_log_prefix('warn', 2)} No {on} data found.")
            return None

        orig_obs = set(adata.uns.get("X_raw_obs_names", []))
        orig_var = set(adata.uns.get("X_raw_var_names", []))
        current_obs = set(adata.obs_names)
        current_var = set(adata.var_names)

        dropped_obs = sorted(list(orig_obs - current_obs))
        dropped_var = sorted(list(orig_var - current_var))

        print(f"   → Samples dropped: {len(dropped_obs)}")
        print(f"   → Features dropped: {len(dropped_var)}")

        return {"dropped_samples": dropped_obs, "dropped_features": dropped_var}

    def show_layer_provenance(
        self,
        layer: str | None = None,
        on: str = "protein",
    ) -> None:
        """
        Pretty-print the provenance chain for one or all layers.

        Uses ``adata.uns['current_X_layer']`` (maintained by ``set_X()``) to highlight
        the active ``.X`` matrix when printing the full registry. Chains walk through
        ``input_layer`` pointers and show non-registry roots (e.g. ``X_raw``) as
        ``(raw input)``.
        """
        adata = utils.get_adata(self, on)  # type: ignore[arg-type]
        registry: dict = adata.uns.get("layer_provenance", {})

        on_label = "protein" if on in ("protein", "prot") else "peptide"

        if not registry:
            print(
                f"{utils.format_log_prefix('info')} No layer provenance recorded yet "
                f"for [{on_label}]. Run normalize(), impute(), or log_transform() first."
            )
            return

        def _build_chain(target: str) -> list[tuple[str, dict | None]]:
            chain: list[tuple[str, dict | None]] = []
            visited: set[str] = set()
            current: str = target
            while current and current not in visited:
                visited.add(current)
                rec = registry.get(current)
                chain.append((current, rec))
                if rec is None:
                    break
                nxt = rec.get("input_layer", "")
                if not nxt:
                    break
                current = nxt
            chain.reverse()
            return chain

        def _print_chain(
            target: str,
            label: str | None = None,
            indent: int = 0,
        ) -> set[str]:
            chain = _build_chain(target)
            shown: set[str] = set()
            pad = "  " * indent
            header = label or f"'{target}'"
            print(f"\n{pad}{header}:")

            for i, (lname, rec) in enumerate(chain):
                shown.add(lname)
                depth_pad = "  " * (indent + i)
                step_num = f"[{i}]"

                if rec is None:
                    print(f"  {depth_pad}{step_num}  '{lname}'  (raw input)")
                else:
                    op = rec.get("op", "?")
                    inp = rec.get("input_layer", "?")
                    extras = {
                        k: v
                        for k, v in rec.items()
                        if k not in ("op", "input_layer")
                    }
                    extras_str = (
                        ", ".join(f"{k}={v}" for k, v in sorted(extras.items()))
                        if extras
                        else ""
                    )
                    tail = f", {extras_str}" if extras_str else ""
                    if i == 0:
                        print(f"  {depth_pad}{step_num}  '{lname}'")
                    else:
                        print(
                            f"  {depth_pad}{step_num}  '{lname}'  ← {op}({inp}{tail})"
                        )

            return shown

        def _find_current_X_layer() -> str | None:
            return adata.uns.get("current_X_layer")

        if layer is not None:
            if layer not in registry:
                print(
                    f"{utils.format_log_prefix('warn')} Layer '{layer}' not found "
                    "in provenance registry."
                )
                return
            _print_chain(layer)
            return

        print(f"  Layer provenance  [{on_label}]")

        shown_layers: set[str] = set()

        current_X = _find_current_X_layer()
        if current_X and current_X in registry:
            print(f"\n  ● Current .X layer:")
            shown_layers |= _print_chain(
                current_X, label=f"  '{current_X}'", indent=1
            )
        elif current_X:
            print(
                f"\n  ● Current .X layer: '{current_X}' (not in registry)"
            )
            shown_layers.add(current_X)

        remaining = sorted(
            lname for lname in registry if lname not in shown_layers
        )
        if remaining:
            print(f"\n  ○ Other layers:")
            for lname in remaining:
                shown_layers |= _print_chain(
                    lname, label=f"  '{lname}'", indent=1
                )

        print(f"\n{'─'*55}\n")