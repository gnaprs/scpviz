/* global window, document */
(function () {
  if (window.__scpvizSvgEditorBridgeInitialized) {
    return;
  }
  window.__scpvizSvgEditorBridgeInitialized = true;

  function fireValueChange(el, value) {
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function editorHtml() {
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0b1325;
      color: #e5e7eb;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }
    #toolbar {
      min-height: 84px;
      padding: 8px;
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
      background: #111c33;
      border-bottom: 1px solid #24324c;
      box-sizing: border-box;
    }
    #toolbar button, #toolbar input {
      border-radius: 8px;
      border: 1px solid #24324c;
      background: #0f172a;
      color: #e5e7eb;
      padding: 6px 8px;
      font-size: 12px;
    }
    #toolbar .hint {
      font-size: 11px;
      color: #94a3b8;
      margin-left: 8px;
    }
    #canvasWrap {
      width: 100%;
      height: calc(100% - 84px);
      overflow: auto;
      padding: 10px;
      box-sizing: border-box;
    }
    #canvas {
      min-width: 100%;
      min-height: 100%;
      background: #ffffff;
      border: 1px solid #24324c;
      cursor: default;
    }
    .sel {
      outline: 2px dashed #2563eb;
      outline-offset: 1px;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="undoBtn" type="button">Undo</button>
    <button id="redoBtn" type="button">Redo</button>
    <button id="addRect" type="button">Rect</button>
    <button id="addCircle" type="button">Circle</button>
    <button id="addText" type="button">Text</button>
    <button id="addLine" type="button">Line</button>
    <button id="duplicateSel" type="button">Duplicate</button>
    <button id="bringFront" type="button">Bring Front</button>
    <button id="sendBack" type="button">Send Back</button>
    <button id="scaleDown" type="button">Scale -</button>
    <button id="scaleUp" type="button">Scale +</button>
    <button id="deleteSel" type="button">Delete</button>
    <label>Fill <input id="fillColor" type="color" value="#60a5fa" /></label>
    <label>Stroke <input id="strokeColor" type="color" value="#1e3a8a" /></label>
    <span class="hint">Arrow keys move; Shift+Arrow moves faster; Ctrl+Z/Y for undo/redo.</span>
  </div>
  <div id="canvasWrap">
    <svg id="canvas" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700"></svg>
  </div>

  <script>
    (function () {
      var ns = "http://www.w3.org/2000/svg";
      var canvas = document.getElementById("canvas");
      var selected = null;
      var dragState = null;
      var undoStack = [];
      var redoStack = [];
      var maxUndo = 80;

      function snapshotState() {
        return {
          viewBox: canvas.getAttribute("viewBox") || "0 0 1200 700",
          inner: canvas.innerHTML
        };
      }

      function restoreSnapshot(state) {
        if (!state) return;
        canvas.setAttribute("viewBox", state.viewBox || "0 0 1200 700");
        canvas.innerHTML = state.inner || "";
        setSelected(null);
      }

      function pushUndo() {
        undoStack.push(snapshotState());
        if (undoStack.length > maxUndo) undoStack.shift();
        redoStack = [];
      }

      function applySnapshotAndNotify(state, dirtyValue) {
        restoreSnapshot(state);
        notify(dirtyValue);
      }

      function undo() {
        if (!undoStack.length) return;
        redoStack.push(snapshotState());
        applySnapshotAndNotify(undoStack.pop(), true);
      }

      function redo() {
        if (!redoStack.length) return;
        undoStack.push(snapshotState());
        applySnapshotAndNotify(redoStack.pop(), true);
      }

      function setSelected(el) {
        if (selected && selected.classList) selected.classList.remove("sel");
        selected = el;
        if (selected && selected.classList) selected.classList.add("sel");
      }

      function safeParentOrigin() {
        try {
          return window.parent.location.origin;
        } catch (e) {
          return "*";
        }
      }

      var PARENT_POST_ORIGIN = safeParentOrigin();

      function notify(dirty) {
        parent.postMessage(
          {
            type: "editorSvgChanged",
            svg: canvas.outerHTML,
            dirty: !!dirty
          },
          PARENT_POST_ORIGIN
        );
      }

      function getPoint(evt) {
        var pt = canvas.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        var ctm = canvas.getScreenCTM();
        if (!ctm) return { x: 0, y: 0 };
        return pt.matrixTransform(ctm.inverse());
      }

      function toNum(v) {
        var out = Number(v);
        return Number.isFinite(out) ? out : 0;
      }

      function moveSelectedBy(dx, dy) {
        if (!selected || selected === canvas) return;
        var tag = (selected.tagName || "").toLowerCase();
        if (tag === "rect" || tag === "image") {
          selected.setAttribute("x", String(toNum(selected.getAttribute("x")) + dx));
          selected.setAttribute("y", String(toNum(selected.getAttribute("y")) + dy));
          return;
        }
        if (tag === "circle") {
          selected.setAttribute("cx", String(toNum(selected.getAttribute("cx")) + dx));
          selected.setAttribute("cy", String(toNum(selected.getAttribute("cy")) + dy));
          return;
        }
        if (tag === "line") {
          selected.setAttribute("x1", String(toNum(selected.getAttribute("x1")) + dx));
          selected.setAttribute("y1", String(toNum(selected.getAttribute("y1")) + dy));
          selected.setAttribute("x2", String(toNum(selected.getAttribute("x2")) + dx));
          selected.setAttribute("y2", String(toNum(selected.getAttribute("y2")) + dy));
          return;
        }
        if (tag === "text") {
          selected.setAttribute("x", String(toNum(selected.getAttribute("x")) + dx));
          selected.setAttribute("y", String(toNum(selected.getAttribute("y")) + dy));
          return;
        }
        var tx = "translate(" + dx + " " + dy + ")";
        var oldTransform = selected.getAttribute("transform") || "";
        selected.setAttribute("transform", (oldTransform + " " + tx).trim());
      }

      function scaleSelected(factor) {
        if (!selected || selected === canvas) return;
        var bbox = selected.getBBox ? selected.getBBox() : null;
        if (!bbox) return;
        var cx = bbox.x + bbox.width / 2;
        var cy = bbox.y + bbox.height / 2;
        var oldTransform = selected.getAttribute("transform") || "";
        var scaleTf = "translate(" + cx + " " + cy + ") scale(" + factor + ") translate(" + (-cx) + " " + (-cy) + ")";
        selected.setAttribute("transform", (oldTransform + " " + scaleTf).trim());
      }

      function startDrag(evt) {
        if (!selected || selected === canvas) return;
        var p = getPoint(evt);
        pushUndo();
        dragState = { startX: p.x, startY: p.y };
      }

      function moveDrag(evt) {
        if (!dragState || !selected) return;
        var p = getPoint(evt);
        var dx = p.x - dragState.startX;
        var dy = p.y - dragState.startY;
        dragState.startX = p.x;
        dragState.startY = p.y;
        moveSelectedBy(dx, dy);
        notify(true);
      }

      function stopDrag() {
        dragState = null;
      }

      function loadSvg(svgText) {
        var cleaned = (svgText || "").trim();
        if (!cleaned) {
          canvas.innerHTML = "";
          setSelected(null);
          undoStack = [];
          redoStack = [];
          notify(false);
          return;
        }
        try {
          var parser = new DOMParser();
          var doc = parser.parseFromString(cleaned, "image/svg+xml");
          var parsed = doc.documentElement;
          if (!parsed || parsed.nodeName.toLowerCase() !== "svg") {
            throw new Error("Invalid SVG payload");
          }
          var viewBox = parsed.getAttribute("viewBox");
          canvas.setAttribute("viewBox", viewBox || "0 0 1200 700");
          canvas.innerHTML = "";
          Array.prototype.slice.call(parsed.childNodes).forEach(function (node) {
            canvas.appendChild(document.importNode(node, true));
          });
          setSelected(null);
          undoStack = [];
          redoStack = [];
          notify(false);
        } catch (err) {
          canvas.innerHTML = "";
          setSelected(null);
          undoStack = [];
          redoStack = [];
          notify(false);
        }
      }

      function applyColors() {
        if (!selected || selected === canvas) return;
        pushUndo();
        var fill = document.getElementById("fillColor").value;
        var stroke = document.getElementById("strokeColor").value;
        selected.setAttribute("fill", fill);
        selected.setAttribute("stroke", stroke);
        selected.setAttribute("stroke-width", selected.getAttribute("stroke-width") || "1.5");
        notify(true);
      }

      function addSvgNode(nodeName, attrs, textContent) {
        pushUndo();
        var el = document.createElementNS(ns, nodeName);
        Object.keys(attrs || {}).forEach(function (k) {
          el.setAttribute(k, String(attrs[k]));
        });
        if (textContent) el.textContent = textContent;
        canvas.appendChild(el);
        setSelected(el);
        notify(true);
      }

      function deleteSelected() {
        if (!selected || selected === canvas) return;
        pushUndo();
        selected.remove();
        setSelected(null);
        notify(true);
      }

      function duplicateSelected() {
        if (!selected || selected === canvas) return;
        pushUndo();
        var clone = selected.cloneNode(true);
        canvas.appendChild(clone);
        setSelected(clone);
        moveSelectedBy(16, 16);
        notify(true);
      }

      function bringToFront() {
        if (!selected || selected === canvas) return;
        pushUndo();
        canvas.appendChild(selected);
        notify(true);
      }

      function sendToBack() {
        if (!selected || selected === canvas) return;
        pushUndo();
        if (canvas.firstChild) {
          canvas.insertBefore(selected, canvas.firstChild);
          notify(true);
        }
      }

      function editTextNode() {
        if (!selected || selected.tagName.toLowerCase() !== "text") return;
        var next = window.prompt("Update text label", selected.textContent || "");
        if (next === null) return;
        pushUndo();
        selected.textContent = next;
        notify(true);
      }

      document.getElementById("undoBtn").addEventListener("click", undo);
      document.getElementById("redoBtn").addEventListener("click", redo);

      document.getElementById("addRect").addEventListener("click", function () {
        addSvgNode("rect", {
          x: 120,
          y: 120,
          width: 180,
          height: 90,
          fill: document.getElementById("fillColor").value,
          stroke: document.getElementById("strokeColor").value,
          "stroke-width": 2
        });
      });

      document.getElementById("addCircle").addEventListener("click", function () {
        addSvgNode("circle", {
          cx: 280,
          cy: 240,
          r: 54,
          fill: document.getElementById("fillColor").value,
          stroke: document.getElementById("strokeColor").value,
          "stroke-width": 2
        });
      });

      document.getElementById("addText").addEventListener("click", function () {
        addSvgNode(
          "text",
          {
            x: 180,
            y: 160,
            fill: document.getElementById("fillColor").value,
            "font-size": 26,
            "font-family": "Arial, sans-serif"
          },
          "Edit label"
        );
      });

      document.getElementById("addLine").addEventListener("click", function () {
        addSvgNode("line", {
          x1: 160,
          y1: 320,
          x2: 360,
          y2: 320,
          stroke: document.getElementById("strokeColor").value,
          "stroke-width": 3
        });
      });

      document.getElementById("duplicateSel").addEventListener("click", duplicateSelected);
      document.getElementById("bringFront").addEventListener("click", bringToFront);
      document.getElementById("sendBack").addEventListener("click", sendToBack);
      document.getElementById("scaleDown").addEventListener("click", function () {
        if (!selected || selected === canvas) return;
        pushUndo();
        scaleSelected(0.92);
        notify(true);
      });
      document.getElementById("scaleUp").addEventListener("click", function () {
        if (!selected || selected === canvas) return;
        pushUndo();
        scaleSelected(1.08);
        notify(true);
      });
      document.getElementById("deleteSel").addEventListener("click", deleteSelected);

      document.getElementById("fillColor").addEventListener("change", applyColors);
      document.getElementById("strokeColor").addEventListener("change", applyColors);

      canvas.addEventListener("mousedown", function (evt) {
        if (evt.target && evt.target !== canvas) {
          setSelected(evt.target);
          startDrag(evt);
        } else {
          setSelected(null);
        }
      });
      canvas.addEventListener("dblclick", function (evt) {
        if (evt.target && evt.target.tagName && evt.target.tagName.toLowerCase() === "text") {
          setSelected(evt.target);
          editTextNode();
        }
      });
      window.addEventListener("mousemove", moveDrag);
      window.addEventListener("mouseup", stopDrag);

      window.addEventListener("keydown", function (evt) {
        var key = evt.key || "";
        var step = evt.shiftKey ? 10 : 2;
        var handled = false;
        if (evt.ctrlKey && (key === "z" || key === "Z")) {
          undo();
          handled = true;
        } else if (evt.ctrlKey && (key === "y" || key === "Y")) {
          redo();
          handled = true;
        } else if (key === "Delete" || key === "Backspace") {
          deleteSelected();
          handled = true;
        } else if (key === "ArrowLeft") {
          if (selected && selected !== canvas) {
            pushUndo();
            moveSelectedBy(-step, 0);
            notify(true);
            handled = true;
          }
        } else if (key === "ArrowRight") {
          if (selected && selected !== canvas) {
            pushUndo();
            moveSelectedBy(step, 0);
            notify(true);
            handled = true;
          }
        } else if (key === "ArrowUp") {
          if (selected && selected !== canvas) {
            pushUndo();
            moveSelectedBy(0, -step);
            notify(true);
            handled = true;
          }
        } else if (key === "ArrowDown") {
          if (selected && selected !== canvas) {
            pushUndo();
            moveSelectedBy(0, step);
            notify(true);
            handled = true;
          }
        }
        if (handled) evt.preventDefault();
      });

      window.addEventListener("message", function (evt) {
        if (evt.origin !== window.location.origin) {
          return;
        }
        var data = evt.data || {};
        if (data.type === "loadSvg") {
          loadSvg(data.svg || "");
        }
      });

      parent.postMessage({ type: "editorReady" }, PARENT_POST_ORIGIN);
      notify(false);
    })();
  </script>
</body>
</html>`;
  }

  function ensureEditorFrame() {
    var iframe = document.getElementById("svg-editor-frame");
    if (!iframe) return null;
    if (!iframe.srcdoc || iframe.srcdoc.indexOf("editorSvgChanged") === -1) {
      iframe.srcdoc = editorHtml();
    }
    return iframe;
  }

  function pushSourceToEditor() {
    var iframe = ensureEditorFrame();
    var sourceEl = document.getElementById("editor-source-svg");
    if (!iframe || !sourceEl || !iframe.contentWindow) return;
    var origin = window.location.origin;
    iframe.contentWindow.postMessage({ type: "loadSvg", svg: sourceEl.value || "" }, origin);
  }

  function setupBridge() {
    ensureEditorFrame();
    var sourceEl = document.getElementById("editor-source-svg");
    if (sourceEl && !sourceEl.__scpvizBound) {
      sourceEl.__scpvizBound = true;
      sourceEl.addEventListener("input", function () {
        pushSourceToEditor();
      });
    }
  }

  window.addEventListener("message", function (evt) {
    if (evt.origin !== window.location.origin) {
      return;
    }
    var iframe = document.getElementById("svg-editor-frame");
    if (!iframe || !iframe.contentWindow || evt.source !== iframe.contentWindow) {
      return;
    }
    var data = evt.data || {};
    if (data.type === "editorReady") {
      pushSourceToEditor();
      return;
    }
    if (data.type !== "editorSvgChanged") return;
    var editedEl = document.getElementById("editor-edited-svg");
    var dirtyEl = document.getElementById("editor-dirty-flag");
    fireValueChange(editedEl, data.svg || "");
    fireValueChange(dirtyEl, data.dirty ? "true" : "false");
  });

  var setupBridgeTimer = null;
  function scheduleSetupBridge() {
    if (setupBridgeTimer) {
      clearTimeout(setupBridgeTimer);
    }
    setupBridgeTimer = setTimeout(function () {
      setupBridgeTimer = null;
      setupBridge();
    }, 150);
  }

  var observer = new MutationObserver(function () {
    scheduleSetupBridge();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  setupBridge();
})();
