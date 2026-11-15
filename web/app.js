"use strict";

(() => {
  const fileInput = document.querySelector("#image-file");
  const scaleInput = document.querySelector("#scale-input");
  const detectButton = document.querySelector("#detect-button");
  const emptyState = document.querySelector("#empty-state");
  const canvasWrap = document.querySelector("#canvas-wrap");
  const canvas = document.querySelector("#result-canvas");
  const context = canvas.getContext("2d");
  const status = document.querySelector("#status");
  const resultMeta = document.querySelector("#result-meta");
  const animalList = document.querySelector("#animal-measurements");
  const pairList = document.querySelector("#pair-measurements");

  let selectedFile = null;
  let sourceImage = null;
  let inference = null;
  let hoverTarget = null;
  let pinnedTarget = null;
  let imageUrl = null;
  let imageReady = false;
  const contourColors = ["#ffff00", "#00ffff", "#ff00ff", "#ffa500"];

  function setStatus(message, kind = "") {
    status.textContent = message;
    status.dataset.kind = kind;
  }

  function formatDistance(value) {
    return value == null ? "未能量測" : `${Number(value).toFixed(2)} mm`;
  }

  function pointFromString(value) {
    if (typeof value !== "string") return null;
    const numbers = value.match(/-?\d+(?:\.\d+)?/g);
    if (!numbers || numbers.length < 2) return null;
    return { x: Number(numbers[0]), y: Number(numbers[1]) };
  }

  function animalKey(animalId) {
    return `animal:${animalId}`;
  }

  function pairKey(pair) {
    return `pair:${pair.animal_1_id}:${pair.animal_2_id}`;
  }

  function activeTarget() {
    return hoverTarget || pinnedTarget;
  }

  function makeCard(target, title, value, detail) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "measurement-card";
    card.dataset.key = target.key;

    const titleNode = document.createElement("span");
    titleNode.className = "card-title";
    titleNode.textContent = title;
    const valueNode = document.createElement("span");
    valueNode.className = "card-value";
    valueNode.textContent = value;
    card.append(titleNode, valueNode);

    if (detail) {
      const detailNode = document.createElement("span");
      detailNode.className = "card-detail";
      detailNode.textContent = detail;
      card.append(detailNode);
    }

    card.addEventListener("mouseenter", () => {
      hoverTarget = target;
      syncActiveCard();
      renderCanvas();
    });
    card.addEventListener("mouseleave", clearHoverTarget);
    card.addEventListener("focus", () => {
      hoverTarget = target;
      syncActiveCard();
      renderCanvas();
    });
    card.addEventListener("blur", clearHoverTarget);
    card.addEventListener("click", () => {
      pinnedTarget = pinnedTarget?.key === target.key ? null : target;
      hoverTarget = null;
      syncActiveCard();
      renderCanvas();
    });
    return card;
  }

  function renderMeasurements() {
    animalList.replaceChildren();
    pairList.replaceChildren();

    const animals = inference?.animals || [];
    const pairs = inference?.pairwise_measurements || [];
    if (!animals.length) {
      animalList.innerHTML = '<p class="empty-list">沒有偵測到動物</p>';
    } else {
      for (const animal of animals) {
        const target = { type: "animal", animalId: animal.animal_id };
        target.key = animalKey(animal.animal_id);
        const title = `${animal.animal_id} · ${animal.class_name}`;
        const candidate = animal.pose_candidate_interocular_dist_mm;
        const official = animal.interocular_dist_mm;
        const displayedDistance = official ?? candidate;
        const hasUnverifiedPoint = [
          animal.pose_candidate_left_eye,
          animal.pose_candidate_right_eye,
        ].some((point) => point && !point.inside_mask);
        const detail = [
          `偵測信心 ${Number(animal.bbox_confidence).toFixed(2)} · 輪廓可互動`,
          official == null && candidate != null
            ? `未驗證 Pose 估計 ${formatDistance(candidate)}`
            : hasUnverifiedPoint && candidate != null
              ? `另有未驗證候選 ${formatDistance(candidate)}`
              : "遮罩內 Pose 估計；眼點語意未經真值驗證",
        ]
          .filter(Boolean)
          .join(" · ");
        animalList.append(
          makeCard(target, title, formatDistance(displayedDistance), detail),
        );
      }
    }

    const displayPairs = pairs.filter(
      (pair) =>
        pair.right_eye_to_right_eye_dist_mm != null ||
        pair.pose_candidate_right_eye_to_right_eye_dist_mm != null,
    );
    if (!displayPairs.length) {
      pairList.innerHTML = '<p class="empty-list">沒有可用的右眼配對距離</p>';
    } else {
      for (const pair of displayPairs) {
        const target = {
          type: "pair",
          animal1Id: pair.animal_1_id,
          animal2Id: pair.animal_2_id,
        };
        target.key = pairKey(pair);
        pairList.append(
          makeCard(
            target,
            `${pair.animal_1_id} ↔ ${pair.animal_2_id}`,
            formatDistance(
              pair.right_eye_to_right_eye_dist_mm ??
                pair.pose_candidate_right_eye_to_right_eye_dist_mm,
            ),
            pair.right_eye_to_right_eye_dist_mm == null
              ? "未驗證 Pose 預估：右眼到右眼"
              : "右眼到右眼（遮罩內估計；眼點語意未驗證）",
          ),
        );
      }
    }

    resultMeta.textContent =
      `${inference.detected_animals_count} 隻動物 · 比例尺 ${Number(inference.pixel_to_mm_scale).toFixed(3)} mm/px`;
    syncActiveCard();
  }

  function syncActiveCard() {
    const activeKey = activeTarget()?.key;
    document.querySelectorAll(".measurement-card").forEach((card) => {
      const active = card.dataset.key === activeKey;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-pressed", String(card.dataset.key === pinnedTarget?.key));
    });
  }

  function animalPolygon(animal) {
    if (Array.isArray(animal.mask_polygon) && animal.mask_polygon.length > 2) {
      return animal.mask_polygon;
    }
    if (Array.isArray(animal.bbox) && animal.bbox.length === 4) {
      const [x1, y1, x2, y2] = animal.bbox;
      return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
    }
    return [];
  }

  function drawPolygon(animal, highlighted, animalIndex) {
    const polygon = animalPolygon(animal);
    if (polygon.length < 3) return;

    context.beginPath();
    context.moveTo(polygon[0][0], polygon[0][1]);
    for (let index = 1; index < polygon.length; index += 1) {
      context.lineTo(polygon[index][0], polygon[index][1]);
    }
    context.closePath();
    const color = contourColors[animalIndex % contourColors.length];
    context.fillStyle = highlighted
      ? "rgba(23, 75, 138, 0.22)"
      : `${color}26`;
    context.fill();
    context.strokeStyle = highlighted ? "#174b8a" : color;
    context.lineWidth = highlighted ? 5 : 2.5;
    context.stroke();
  }

  function drawLine(start, end, color, width, dashed = false) {
    if (!start || !end) return;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = color;
    context.lineWidth = width;
    context.lineCap = "round";
    context.setLineDash(dashed ? [7, 5] : []);
    context.stroke();
    context.setLineDash([]);
  }

  function drawPoseCandidate(point) {
    if (!point || point.x == null || point.y == null) return;
    context.beginPath();
    if (point.inside_mask) {
      context.arc(point.x, point.y, 6, 0, Math.PI * 2);
      context.fillStyle = "#00dc00";
      context.fill();
      context.strokeStyle = "#fff";
      context.lineWidth = 1.5;
      context.stroke();
    } else {
      context.moveTo(point.x - 6, point.y - 6);
      context.lineTo(point.x + 6, point.y + 6);
      context.moveTo(point.x + 6, point.y - 6);
      context.lineTo(point.x - 6, point.y + 6);
      context.strokeStyle = "#ff0000";
      context.lineWidth = 2.5;
      context.stroke();
    }
  }

  function renderCanvas() {
    if (!sourceImage || !inference) return;
    const target = activeTarget();
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

    (inference.animals || []).forEach((animal, index) => {
      drawPolygon(
        animal,
        target?.type === "animal" && target.animalId === animal.animal_id,
        index,
      );
    });

    for (const animal of inference.animals || []) {
      const candidateLeft = animal.pose_candidate_left_eye;
      const candidateRight = animal.pose_candidate_right_eye;
      if (
        animal.interocular_dist_mm == null &&
        candidateLeft &&
        candidateRight
      ) {
        const selected = target?.type === "animal" && target.animalId === animal.animal_id;
        const lineSelected = target?.type === "eye" && target.animalId === animal.animal_id;
        drawLine(
          candidateLeft,
          candidateRight,
          selected || lineSelected ? "#123b70" : "#0050ff",
          selected || lineSelected ? 4 : 2,
          true,
        );
      }
      if (!animal.left_eye || !animal.right_eye) continue;
      const selected = target?.type === "animal" && target.animalId === animal.animal_id;
      const lineSelected = target?.type === "eye" && target.animalId === animal.animal_id;
      drawLine(
        animal.left_eye,
        animal.right_eye,
        selected || lineSelected ? "#123b70" : "#ffff00",
        selected || lineSelected ? 5 : 2.5,
      );
    }

    for (const pair of inference.pairwise_measurements || []) {
      const candidate = pair.right_eye_to_right_eye_dist_mm == null;
      if (
        candidate &&
        pair.pose_candidate_right_eye_to_right_eye_dist_mm == null
      ) continue;
      const start = candidate
        ? pair.pose_candidate_animal_1_right_eye
        : pointFromString(pair.animal_1_right_eye);
      const end = candidate
        ? pair.pose_candidate_animal_2_right_eye
        : pointFromString(pair.animal_2_right_eye);
      const selected = target?.type === "pair" && pairKey(pair) === target.key;
      drawLine(
        start,
        end,
        selected ? "#123b70" : candidate ? "#0050ff" : "#ff00ff",
        selected ? 5 : 2,
        candidate,
      );
    }

    for (const animal of inference.animals || []) {
      if (animal.pose_candidate_left_eye || animal.pose_candidate_right_eye) {
        drawPoseCandidate(animal.pose_candidate_left_eye);
        drawPoseCandidate(animal.pose_candidate_right_eye);
        continue;
      }
      for (const [eye, color] of [
        [animal.left_eye, "#1769d2"],
        [animal.right_eye, "#e25835"],
      ]) {
        if (!eye || eye.x == null || eye.y == null) continue;
        context.beginPath();
        context.arc(eye.x, eye.y, 5, 0, Math.PI * 2);
        context.fillStyle = color;
        context.fill();
        context.strokeStyle = "#fff";
        context.lineWidth = 1.5;
        context.stroke();
      }
    }
  }

  function distanceToSegment(point, start, end) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    if (!lengthSquared) return Math.hypot(point.x - start.x, point.y - start.y);
    const projection = Math.max(
      0,
      Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared),
    );
    return Math.hypot(
      point.x - (start.x + projection * dx),
      point.y - (start.y + projection * dy),
    );
  }

  function nearLine(point, start, end, threshold) {
    return start && end && distanceToSegment(point, start, end) <= threshold;
  }

  function nearOutline(point, animal, threshold) {
    const polygon = animalPolygon(animal);
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const next = polygon[(index + 1) % polygon.length];
      if (nearLine(point, { x: current[0], y: current[1] }, { x: next[0], y: next[1] }, threshold)) {
        return true;
      }
    }
    return false;
  }

  function hitTest(event) {
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / canvas.width;
    if (!scale) return null;
    const point = {
      x: (event.clientX - rect.left) / scale,
      y: (event.clientY - rect.top) / scale,
    };
    const threshold = 9 / scale;

    for (const pair of inference.pairwise_measurements || []) {
      const candidate = pair.right_eye_to_right_eye_dist_mm == null;
      if (
        candidate &&
        pair.pose_candidate_right_eye_to_right_eye_dist_mm == null
      ) continue;
      const start = candidate
        ? pair.pose_candidate_animal_1_right_eye
        : pointFromString(pair.animal_1_right_eye);
      const end = candidate
        ? pair.pose_candidate_animal_2_right_eye
        : pointFromString(pair.animal_2_right_eye);
      if (nearLine(point, start, end, threshold)) {
        const target = {
          type: "pair",
          animal1Id: pair.animal_1_id,
          animal2Id: pair.animal_2_id,
        };
        target.key = pairKey(pair);
        return target;
      }
    }

    for (const animal of inference.animals || []) {
      const validated = animal.interocular_dist_mm != null;
      const start = validated
        ? animal.left_eye
        : animal.pose_candidate_left_eye;
      const end = validated
        ? animal.right_eye
        : animal.pose_candidate_right_eye;
      if (start && end && nearLine(point, start, end, threshold)) {
        const target = { type: "eye", animalId: animal.animal_id };
        target.key = animalKey(animal.animal_id);
        return target;
      }
    }

    for (const animal of inference.animals || []) {
      if (nearOutline(point, animal, threshold)) {
        const target = { type: "animal", animalId: animal.animal_id };
        target.key = animalKey(animal.animal_id);
        return target;
      }
    }
    return null;
  }

  function clearHoverTarget() {
    hoverTarget = null;
    syncActiveCard();
    renderCanvas();
  }

  fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files?.[0] || null;
    inference = null;
    sourceImage = null;
    imageReady = false;
    hoverTarget = null;
    pinnedTarget = null;
    detectButton.disabled = !selectedFile;
    canvasWrap.hidden = true;
    emptyState.hidden = false;
    animalList.innerHTML = '<p class="empty-list">尚無量測資料</p>';
    pairList.innerHTML = '<p class="empty-list">尚無量測資料</p>';
    resultMeta.textContent = "偵測結果會顯示在這裡";
    setStatus("");
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    if (!selectedFile) return;

    imageUrl = URL.createObjectURL(selectedFile);
    sourceImage = new Image();
    sourceImage.onload = () => {
      canvas.width = sourceImage.naturalWidth;
      canvas.height = sourceImage.naturalHeight;
      imageReady = true;
      detectButton.disabled = false;
      setStatus(`已載入 ${selectedFile.name}（${canvas.width} × ${canvas.height}）`);
    };
    sourceImage.onerror = () => {
      imageReady = false;
      detectButton.disabled = true;
      setStatus("無法讀取這張圖片。", "error");
    };
    sourceImage.src = imageUrl;
  });

  detectButton.addEventListener("click", async () => {
    if (!selectedFile || !imageReady) return;
    const scale = Number(scaleInput.value);
    if (!Number.isFinite(scale) || scale <= 0) {
      setStatus("比例尺必須是大於 0 的數值。", "error");
      scaleInput.focus();
      return;
    }

    detectButton.disabled = true;
    setStatus("模型推論中，請稍候…");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const query = new URLSearchParams({
        pixel_to_mm_scale: String(scale),
        save_to_csv: "false",
      });
      const response = await fetch(`/predict?${query}`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `偵測失敗（HTTP ${response.status}）`);
      }

      inference = await response.json();
      canvas.width = sourceImage.naturalWidth;
      canvas.height = sourceImage.naturalHeight;
      hoverTarget = null;
      pinnedTarget = null;
      emptyState.hidden = true;
      canvasWrap.hidden = false;
      renderMeasurements();
      renderCanvas();
      setStatus(`完成：找到 ${inference.detected_animals_count} 隻動物。`, "success");
    } catch (error) {
      setStatus(error.message || "偵測失敗，請確認 API 與模型已啟動。", "error");
    } finally {
      detectButton.disabled = !imageReady;
    }
  });

  canvas.addEventListener("mousemove", (event) => {
    if (!inference) return;
    const nextTarget = hitTest(event);
    if (nextTarget?.key === hoverTarget?.key) return;
    hoverTarget = nextTarget;
    syncActiveCard();
    renderCanvas();
  });

  canvas.addEventListener("mouseleave", clearHoverTarget);
})();
