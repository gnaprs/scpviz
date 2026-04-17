/* global window, document */
(function () {
  if (window.__scpvizDeColorPickerInitialized) {
    return;
  }
  window.__scpvizDeColorPickerInitialized = true;

  var buttonToInputMap = {
    "btn-pick-de-color-up": "de-color-up",
    "btn-pick-de-color-down": "de-color-down",
    "btn-pick-de-color-ns": "de-color-ns",
    "btn-pick-de-highlight-color": "de-highlight-labeled-color"
  };

  function normalizeHex(value, fallback) {
    var text = String(value || "").trim();
    if (/^#[0-9a-fA-F]{6}$/.test(text)) return text;
    return fallback || "#16a34a";
  }

  function setDashInputValue(inputEl, value) {
    if (!inputEl) return;
    inputEl.value = value;
    inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    inputEl.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function openNativePicker(targetInputId) {
    var targetInput = document.getElementById(targetInputId);
    if (!targetInput) return;
    var picker = document.createElement("input");
    picker.type = "color";
    picker.value = normalizeHex(targetInput.value, "#16a34a");
    picker.style.position = "fixed";
    picker.style.left = "-9999px";
    picker.style.top = "-9999px";
    picker.style.opacity = "0";
    document.body.appendChild(picker);

    picker.addEventListener("input", function () {
      setDashInputValue(targetInput, picker.value);
    });
    picker.addEventListener("change", function () {
      setDashInputValue(targetInput, picker.value);
      if (picker.parentNode) picker.parentNode.removeChild(picker);
    });
    picker.addEventListener("blur", function () {
      if (picker.parentNode) picker.parentNode.removeChild(picker);
    });
    try {
      if (typeof picker.showPicker === "function") {
        picker.showPicker();
      } else {
        picker.focus();
        picker.click();
      }
    } catch (_err) {
      picker.focus();
      picker.click();
    }
  }

  function bindButtons() {
    Object.keys(buttonToInputMap).forEach(function (buttonId) {
      var button = document.getElementById(buttonId);
      if (!button || button.__deColorPickerBound) return;
      button.__deColorPickerBound = true;
      button.__deColorPickerSuppressClick = false;
      button.addEventListener("pointerdown", function (event) {
        event.preventDefault();
        event.stopPropagation();
        button.__deColorPickerSuppressClick = true;
        openNativePicker(buttonToInputMap[buttonId]);
      });
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (button.__deColorPickerSuppressClick) {
          button.__deColorPickerSuppressClick = false;
          return;
        }
        openNativePicker(buttonToInputMap[buttonId]);
      });
    });
  }

  var observer = new MutationObserver(function () {
    bindButtons();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  bindButtons();
})();
