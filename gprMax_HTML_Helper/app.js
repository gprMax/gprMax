const C0 = 299792458;

const state = {
  materials: [],
  poles: [],
  geometry: [],
  waveforms: [],
  sources: [],
  receivers: [],
  snapshots: [],
  other: [],
  ids: { material: 1, pole: 1, geometry: 1, waveform: 1, source: 1, rx: 1, snapshot: 1 },
};

const $ = (id) => document.getElementById(id);
const value = (id) => $(id).value.trim();
const num = (id) => Number.parseFloat(value(id)) || 0;

const lengthUnits = ["m", "mm", "um", "nm"];
const geometrySchemas = {
  edge: {
    description: "#edge: start point and end point, then material.",
    material: true,
    smoothing: false,
    fields: coords("start", ["x1", "y1", "z1"]).concat(coords("end", ["x2", "y2", "z2"])),
    build: (v, material) => line("#edge", [...vals(v, "x1", "y1", "z1", "x2", "y2", "z2"), material]),
  },
  plate: {
    description: "#plate: lower-left point and upper-right point defining a surface, then material.",
    material: true,
    smoothing: false,
    fields: coords("lower left", ["x1", "y1", "z1"]).concat(coords("upper right", ["x2", "y2", "z2"])),
    build: (v, material) => line("#plate", [...vals(v, "x1", "y1", "z1", "x2", "y2", "z2"), material]),
  },
  triangle: {
    description: "#triangle: three apex points, prism thickness, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("apex 1", ["x1", "y1", "z1"]).concat(coords("apex 2", ["x2", "y2", "z2"]), coords("apex 3", ["x3", "y3", "z3"]), [lengthField("thickness", "thickness", "0")]),
    build: (v, material, smoothing) => line("#triangle", [...vals(v, "x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "thickness"), material, smoothing]),
  },
  box: {
    description: "#box: lower-left and upper-right corners, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("min", ["xmin", "ymin", "zmin"]).concat(coords("max", ["xmax", "ymax", "zmax"], ["0.24", "0.18", "0.002"])),
    build: (v, material, smoothing) => line("#box", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), material, smoothing]),
  },
  sphere: {
    description: "#sphere: centre point, radius, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("centre", ["x", "y", "z"], ["0.12", "0.09", "0.001"]).concat([lengthField("radius", "radius", "0.01")]),
    build: (v, material, smoothing) => line("#sphere", [...vals(v, "x", "y", "z", "radius"), material, smoothing]),
  },
  cylinder: {
    description: "#cylinder: centres of two faces, radius, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("face 1", ["x1", "y1", "z1"], ["0.02", "0.09", "0.001"]).concat(coords("face 2", ["x2", "y2", "z2"], ["0.22", "0.09", "0.001"]), [lengthField("radius", "radius", "0.01")]),
    build: (v, material, smoothing) => line("#cylinder", [...vals(v, "x1", "y1", "z1", "x2", "y2", "z2", "radius"), material, smoothing]),
  },
  cylindrical_sector: {
    description: "#cylindrical_sector: axis, centre coordinates in cross-section, axis min/max, radius, start angle, swept angle, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: [
      selectField("axis", "axis", ["x", "y", "z"], "z"),
      lengthField("centre coord 1", "c1", "0.34"),
      lengthField("centre coord 2", "c2", "0.24"),
      lengthField("axis min", "a1", "0.5"),
      lengthField("axis max", "a2", "0.502"),
      lengthField("radius", "radius", "0.25"),
      textField("start angle deg", "start", "330"),
      textField("swept angle deg", "sweep", "60"),
    ],
    build: (v, material, smoothing) => line("#cylindrical_sector", [v.axis, ...vals(v, "c1", "c2", "a1", "a2", "radius"), v.start, v.sweep, material, smoothing]),
  },
  cone: {
    description: "#cone: centres of two faces, radius at each face, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("face 1", ["x1", "y1", "z1"]).concat(coords("face 2", ["x2", "y2", "z2"], ["0.08", "0.075", "0.075"]), [lengthField("radius 1", "r1", "0.03"), lengthField("radius 2", "r2", "0")]),
    build: (v, material, smoothing) => line("#cone", [...vals(v, "x1", "y1", "z1", "x2", "y2", "z2", "r1", "r2"), material, smoothing]),
  },
  ellipsoid: {
    description: "#ellipsoid: centre point, semi-axis lengths, material, optional smoothing.",
    material: true,
    smoothing: true,
    fields: coords("centre", ["x", "y", "z"], ["0.045", "0.045", "0.045"]).concat(coords("semi-axis", ["sx", "sy", "sz"], ["0.03", "0.02", "0.03"])),
    build: (v, material, smoothing) => line("#ellipsoid", [...vals(v, "x", "y", "z", "sx", "sy", "sz"), material, smoothing]),
  },
  fractal_box: {
    description: "#fractal_box: bounds, fractal parameters, material/mixing model id, fractal id, optional seed and smoothing.",
    material: false,
    smoothing: true,
    fields: coords("min", ["xmin", "ymin", "zmin"]).concat(coords("max", ["xmax", "ymax", "zmax"], ["0.1", "0.1", "0.1"]), [
      textField("fractal dimension", "dimension", "1.5"),
      textField("x weight", "wx", "1"),
      textField("y weight", "wy", "1"),
      textField("z weight", "wz", "1"),
      textField("material count", "count", "1"),
      textField("mixing/material id", "materialRef", "soil"),
      textField("fractal box id", "boxId", "my_fractal_box"),
      textField("seed optional", "seed", ""),
    ]),
    build: (v, _material, smoothing) => line("#fractal_box", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), v.dimension, v.wx, v.wy, v.wz, v.count, v.materialRef, v.boxId, v.seed, smoothing]),
  },
  add_surface_roughness: {
    description: "#add_surface_roughness: surface bounds on a fractal box, fractal settings, lower/upper roughness, fractal box id, optional seed.",
    material: false,
    smoothing: false,
    fields: coords("surface min", ["xmin", "ymin", "zmin"]).concat(coords("surface max", ["xmax", "ymax", "zmax"], ["0.1", "0.1", "0.1"]), [
      textField("fractal dimension", "dimension", "1.5"),
      textField("weight 1", "w1", "1"),
      textField("weight 2", "w2", "1"),
      lengthField("lower roughness", "lower", "0.085"),
      lengthField("upper roughness", "upper", "0.11"),
      textField("fractal box id", "boxId", "my_fractal_box"),
      textField("seed optional", "seed", ""),
    ]),
    build: (v) => line("#add_surface_roughness", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), v.dimension, v.w1, v.w2, ...vals(v, "lower", "upper"), v.boxId, v.seed]),
  },
  add_surface_water: {
    description: "#add_surface_water: surface bounds on a fractal box, water depth, fractal box id.",
    material: false,
    smoothing: false,
    fields: coords("surface min", ["xmin", "ymin", "zmin"]).concat(coords("surface max", ["xmax", "ymax", "zmax"], ["0.1", "0.1", "0.1"]), [lengthField("water depth", "depth", "0.005"), textField("fractal box id", "boxId", "my_fractal_box")]),
    build: (v) => line("#add_surface_water", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "depth"), v.boxId]),
  },
  add_grass: {
    description: "#add_grass: surface bounds on a fractal box, fractal dimension, grass height range, blade count, fractal box id, optional seed.",
    material: false,
    smoothing: false,
    fields: coords("surface min", ["xmin", "ymin", "zmin"]).concat(coords("surface max", ["xmax", "ymax", "zmax"], ["0.1", "0.1", "0.1"]), [
      textField("fractal dimension", "dimension", "1.5"),
      lengthField("min height", "hmin", "0.1"),
      lengthField("max height", "hmax", "0.15"),
      textField("blade count", "count", "100"),
      textField("fractal box id", "boxId", "my_fractal_box"),
      textField("seed optional", "seed", ""),
    ]),
    build: (v) => line("#add_grass", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), v.dimension, ...vals(v, "hmin", "hmax"), v.count, v.boxId, v.seed]),
  },
  geometry_objects_read: {
    description: "#geometry_objects_read: lower-left insertion point, HDF5 file, material text file.",
    material: false,
    smoothing: false,
    fields: coords("insert at", ["x", "y", "z"]).concat([textField("HDF5 file", "file1", "geometry.h5"), textField("materials file", "file2", "materials.txt")]),
    build: (v) => line("#geometry_objects_read", [...vals(v, "x", "y", "z"), v.file1, v.file2]),
  },
  geometry_objects_write: {
    description: "#geometry_objects_write: bounds to save and output basename.",
    material: false,
    smoothing: false,
    fields: coords("min", ["xmin", "ymin", "zmin"]).concat(coords("max", ["xmax", "ymax", "zmax"], ["0.24", "0.18", "0.002"]), [textField("basename", "basename", "geometry")]),
    build: (v) => line("#geometry_objects_write", [...vals(v, "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), v.basename]),
  },
};

function coords(prefix, names, defaults = []) {
  return names.map((name, index) => lengthField(`${prefix} ${name}`, name, defaults[index] ?? "0"));
}

function lengthField(label, key, initial) {
  return { type: "length", label, key, initial };
}

function textField(label, key, initial) {
  return { type: "text", label, key, initial };
}

function selectField(label, key, options, initial) {
  return { type: "select", label, key, options, initial };
}

function fmt(value) {
  if (!Number.isFinite(value) || value === 0) return "0";
  if (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e4) return value.toExponential(6).replace("e+", "e");
  return Number(value.toFixed(6)).toString();
}

function convertLength(v, unit) {
  return v * ({ m: 1, mm: 1e-3, um: 1e-6, nm: 1e-9 }[unit] ?? 1);
}

function convertFrequency(v, unit) {
  return v * ({ Hz: 1, kHz: 1e3, MHz: 1e6, GHz: 1e9, THz: 1e12 }[unit] ?? 1);
}

function convertTime(v, unit) {
  return v * ({ s: 1, ms: 1e-3, us: 1e-6, ns: 1e-9, ps: 1e-12 }[unit] ?? 1);
}

function fromBaseLength(v, unit) {
  return v / ({ m: 1, mm: 1e-3, um: 1e-6, nm: 1e-9 }[unit] ?? 1);
}

function fromBaseTime(v, unit) {
  return v / ({ s: 1, ms: 1e-3, us: 1e-6, ns: 1e-9, ps: 1e-12 }[unit] ?? 1);
}

function convertLengthField(key) {
  return fmt(convertLength(Number.parseFloat(value(`geom_${key}`)) || 0, value(`geom_${key}_unit`)));
}

function vals(v, ...keys) {
  return keys.map((key) => v[key]);
}

function line(command, parts) {
  return `${command}: ${parts.filter((part) => String(part ?? "").trim() !== "").join(" ")}`;
}

function materialNames() {
  return state.materials.map((m) => m.name);
}

function addMaterial() {
  const name = value("materialId");
  if (!name) return setStatus("Material identifier cannot be empty.");
  const existing = state.materials.find((m) => m.name === name);
  const material = {
    id: existing?.id ?? `m${state.ids.material++}`,
    name,
    command: `#material: ${value("materialEps") || "1"} ${value("materialSigma") || "0"} ${value("materialMu") || "1"} ${value("materialSigmaStar") || "0"} ${name}`,
  };
  if (existing) Object.assign(existing, material);
  else state.materials.push(material);
  renderAll();
}

function upsertMaterial(name, eps, sigma, mu, loss) {
  const existing = state.materials.find((m) => m.name === name);
  const material = {
    id: existing?.id ?? `m${state.ids.material++}`,
    name,
    command: `#material: ${eps || "1"} ${sigma || "0"} ${mu || "1"} ${loss || "0"} ${name}`,
  };
  if (existing) Object.assign(existing, material);
  else state.materials.push(material);
}

function addAnisotropicMaterials() {
  const prefix = value("anisPrefix");
  if (!prefix) return setStatus("Anisotropic material prefix cannot be empty.");
  upsertMaterial(`${prefix}_x`, value("anisEpsX"), value("anisSigmaX"), value("anisMuX"), value("anisLossX"));
  upsertMaterial(`${prefix}_y`, value("anisEpsY"), value("anisSigmaY"), value("anisMuY"), value("anisLossY"));
  upsertMaterial(`${prefix}_z`, value("anisEpsZ"), value("anisSigmaZ"), value("anisMuZ"), value("anisLossZ"));
  renderAll();
}

function removeMaterial(id) {
  const material = state.materials.find((m) => m.id === id);
  if (!material) return;
  const dependentPoles = state.poles.filter((p) => p.material === material.name).length;
  const dependentGeometry = state.geometry.filter((g) => String(g.material).split(/\s+/).includes(material.name)).length;
  if ((dependentPoles || dependentGeometry) && !window.confirm(`Removing material '${material.name}' will also remove ${dependentPoles} dispersion command(s) and ${dependentGeometry} geometry command(s). Continue?`)) return;
  state.materials = state.materials.filter((m) => m.id !== id);
  state.poles = state.poles.filter((p) => p.material !== material.name);
  state.geometry = state.geometry.filter((g) => !String(g.material).split(/\s+/).includes(material.name));
  renderAll();
}

function addPole() {
  const material = value("dispMaterial");
  if (!material) return setStatus("Add a material before adding dispersion.");
  const type = value("dispType");
  if (value("dispMode") === "Single pole") state.poles = state.poles.filter((p) => !(p.material === material && p.type === type));
  state.poles.push({
    id: `p${state.ids.pole++}`,
    material,
    type,
    a: value("dispA") || "0",
    freq: convertDispersionFrequency(),
    damping: convertFrequency(num("dispDamping"), value("dispDampingUnit")),
  });
  renderAll();
}

function convertDispersionFrequency() {
  const raw = Number.parseFloat(value("dispFreq")) || 0;
  const unit = value("dispFreqUnit");
  if (["s", "ms", "us", "ns", "ps"].includes(unit)) return fmt(convertTime(raw, unit));
  return fmt(convertFrequency(raw, unit));
}

function poleCommand(pole) {
  if (pole.command) return pole.command;
  if (pole.type === "Debye") return `#add_dispersion_debye: 1 ${pole.a} ${pole.freq} ${pole.material}`;
  if (pole.type === "Lorentz") return `#add_dispersion_lorentz: 1 ${pole.a} ${pole.freq} ${fmt(pole.damping)} ${pole.material}`;
  return `#add_dispersion_drude: 1 ${pole.freq} ${pole.a} ${pole.material}`;
}

function updateDispersionText() {
  const type = value("dispType");
  $("dispAName").firstChild.nodeValue = type === "Drude" ? "Gamma" : "Delta epsilon";
  $("dispFreqName").firstChild.nodeValue = type === "Debye" ? "Relaxation time" : "Pole frequency";
  $("dispDampingWrap").style.display = type === "Lorentz" ? "grid" : "none";
  $("dispDescription").innerHTML = {
    Debye: "Debye uses \\(\\chi_p(t)=\\frac{\\Delta\\epsilon_{rp}}{\\tau_p}e^{-t/\\tau_p}\\). Command values are pole count, \\(\\Delta\\epsilon_{rp}=\\epsilon_{rsp}-\\epsilon_{r\\infty}\\), relaxation time \\(\\tau_p\\) in seconds, then material. The base material \\(\\epsilon_r\\) should be \\(\\epsilon_{r\\infty}\\).",
    Lorentz: "Lorentz uses \\(\\chi_p(t)=\\Re\\{-j\\gamma_p e^{(-\\delta_p+j\\beta_p)t}\\}\\), with \\(\\beta_p=\\sqrt{\\omega_p^2-\\delta_p^2}\\) and \\(\\gamma_p=\\omega_p^2\\Delta\\epsilon_{rp}/\\beta_p\\). Command values are pole count, \\(\\Delta\\epsilon_{rp}\\), pole frequency \\(\\omega_p\\) in Hz, damping \\(\\delta_p\\) in Hz, then material.",
    Drude: "Drude uses \\(\\chi_p(t)=\\frac{\\omega_p^2}{\\gamma_p}(1-e^{-\\gamma_p t})\\). Command values are pole count, pole frequency \\(\\omega_p\\) in Hz, inverse relaxation time \\(\\gamma_p\\) in Hz, then material.",
  }[type];
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([$("dispDescription")]);
}

function domainValues() {
  return {
    x: convertLength(num("domainX"), value("domainXUnit")),
    y: convertLength(num("domainY"), value("domainYUnit")),
    z: convertLength(num("domainZ"), value("domainZUnit")),
    fmax: convertFrequency(num("maxFreq"), value("maxFreqUnit")),
    eps: Math.max(num("epsMax"), 1),
  };
}

function dxSuggestionSet() {
  const d = domainValues();
  const wavelength = C0 / Math.max(d.fmax, 1) / Math.sqrt(d.eps);
  const values = [6, 8, 10, 15, 20].map((divisor) => wavelength / divisor);
  const labels = ["very coarse", "coarse", "normal", "fine", "extra fine"];
  const feature = convertLength(num("smallestFeature"), value("featureUnit"));
  let recommended = 2;
  const reasons = [`normal gives about 10 cells per shortest wavelength in eps_r=${fmt(d.eps)}`];
  if (feature > 0 && feature < values[2] * 3) {
    recommended = Math.max(recommended, 3);
    reasons.push("small features are near only a few normal cells");
  }
  if ($("curvedShape").checked) {
    recommended = Math.max(recommended, 3);
    reasons.push("curved geometry benefits from finer staircasing");
  }
  if ($("subpixel").checked) {
    recommended = Math.max(recommended, 3);
    reasons.push("subpixel smoothing is more useful with a fine grid");
  }
  if (feature > 0 && feature < values[3] * 2) {
    recommended = 4;
    reasons.push("the smallest feature justifies extra fine cells");
  }
  const text = values.map((v, i) => `${i === recommended ? ">>> " : "    "}${labels[i].padEnd(12)} dx=dy=dz=${fmt(v)} m`).join("\n");
  return { values, recommended, text: `${text}\n\nReason: ${reasons.join("; ")}.` };
}

function timeSuggestionSet() {
  const d = domainValues();
  const speed = C0 / Math.sqrt(d.eps);
  const base = Math.max(Math.max(d.x, d.y, d.z, 0.001) / speed, 6 / Math.max(d.fmax, 1));
  const values = [2, 4, 8, 12, 20].map((multiplier) => base * multiplier);
  const labels = ["very low resonance", "low resonance", "normal", "resonant", "very resonant"];
  let recommended = 2;
  const reasons = ["normal covers several source cycles and multiple passes across the largest domain dimension"];
  if ($("highEps").checked || d.eps >= 10) {
    recommended = 3;
    reasons.push("high refractive index slows propagation and can store energy longer");
  }
  if (d.eps >= 25) {
    recommended = 4;
    reasons.push("very high refractive index may produce stronger late-time ringing");
  }
  if ($("curvedShape").checked) {
    recommended = Math.max(recommended, 3);
    reasons.push("curved inclusions can increase scattering and late arrivals");
  }
  const text = values.map((v, i) => `${i === recommended ? ">>> " : "    "}${labels[i].padEnd(18)} time_window=${fmt(v)} s`).join("\n");
  return { values, recommended, text: `${text}\n\nReason: ${reasons.join("; ")}.` };
}

function applyDxChoice() {
  const suggestions = dxSuggestionSet();
  const map = { "Very coarse": 0, Coarse: 1, Normal: 2, Fine: 3, "Extra fine": 4, Recommended: suggestions.recommended };
  const index = map[value("dxChoice")];
  if (index !== undefined) {
    const display = fmt(fromBaseLength(suggestions.values[index], value("dxUnit")));
    $("dxX").value = display;
    $("dxY").value = display;
    $("dxZ").value = display;
  }
}

function applyTimeChoice() {
  const suggestions = timeSuggestionSet();
  const map = { "Very low resonance": 0, "Low resonance": 1, Normal: 2, Resonant: 3, "Very resonant": 4, Recommended: suggestions.recommended };
  const index = map[value("timeChoice")];
  if (index !== undefined) {
    $("timeWindow").value = fmt(fromBaseTime(suggestions.values[index], value("timeUnit")));
  }
}

function modelCommands() {
  const d = domainValues();
  const dx = convertLength(num("dxX"), value("dxUnit"));
  const dy = convertLength(num("dxY"), value("dxUnit"));
  const dz = convertLength(num("dxZ"), value("dxUnit"));
  const tw = convertTime(num("timeWindow"), value("timeUnit"));
  return [`#title: ${value("titleValue") || "model"}`, `#domain: ${fmt(d.x)} ${fmt(d.y)} ${fmt(d.z)}`, `#dx_dy_dz: ${fmt(dx)} ${fmt(dy)} ${fmt(dz)}`, `#time_window: ${fmt(tw)}`].join("\n");
}

function renderGeometryFields() {
  const select = $("shapeType");
  if (!select.options.length) {
    for (const name of Object.keys(geometrySchemas)) select.add(new Option(name, name));
  }
  const schema = geometrySchemas[value("shapeType") || "box"];
  $("shapeDescription").textContent = schema.description;
  $("shapeMaterialWrap").style.display = schema.material ? "grid" : "none";
  $("shapeSmoothingWrap").style.display = schema.smoothing ? "grid" : "none";
  $("anisotropyControls").style.display = schema.material && schema.smoothing ? "grid" : "none";
  const container = $("geometryFields");
  container.replaceChildren();
  for (const field of schema.fields) container.appendChild(createGeometryField(field));
}

function createGeometryField(field) {
  const label = document.createElement("label");
  label.textContent = field.label;
  if (field.type === "select") {
    const select = document.createElement("select");
    select.id = `geom_${field.key}`;
    for (const option of field.options) select.add(new Option(option, option));
    select.value = field.initial;
    label.appendChild(select);
    return label;
  }
  if (field.type === "length") {
    const wrap = document.createElement("div");
    wrap.className = "inline-unit";
    const input = document.createElement("input");
    input.id = `geom_${field.key}`;
    input.value = fmt(fromBaseLength(Number.parseFloat(field.initial) || 0, "mm"));
    const unit = document.createElement("select");
    unit.id = `geom_${field.key}_unit`;
    for (const option of lengthUnits) unit.add(new Option(option, option));
    unit.value = "mm";
    wrap.append(input, unit);
    label.appendChild(wrap);
    return label;
  }
  const input = document.createElement("input");
  input.id = `geom_${field.key}`;
  input.value = field.initial;
  label.appendChild(input);
  return label;
}

function readGeometryValues(schema) {
  const values = {};
  for (const field of schema.fields) {
    if (field.type === "length") values[field.key] = convertLengthField(field.key);
    else values[field.key] = value(`geom_${field.key}`);
  }
  return values;
}

function addGeometry() {
  const type = value("shapeType");
  const schema = geometrySchemas[type];
  let material = schema.material ? value("shapeMaterial") : "";
  if (schema.material && value("shapeMaterialMode") === "Anisotropic x/y/z" && schema.smoothing) {
    const mx = value("shapeMaterialX");
    const my = value("shapeMaterialY");
    const mz = value("shapeMaterialZ");
    if (!mx || !my || !mz) return setStatus("Choose x, y, and z materials for anisotropic geometry.");
    material = `${mx} ${my} ${mz}`;
  }
  if (schema.material && !material) return setStatus("Add a material before using this geometry command.");
  const smoothing = schema.smoothing ? value("shapeSmoothing") : "";
  const values = readGeometryValues(schema);
  const command = schema.build(values, material, smoothing);
  const materialDependency = material || (materialNames().includes(values.materialRef) ? values.materialRef : "");
  state.geometry.push({ id: `g${state.ids.geometry++}`, type, material: materialDependency, command });
  renderAll();
}

function addWaveform() {
  const freq = convertFrequency(num("waveformFreq"), value("waveformFreqUnit"));
  const waveId = value("waveformId") || `pulse${state.ids.waveform}`;
  const waveform = `#waveform: ${value("waveformType")} ${value("waveformAmp") || "1"} ${fmt(freq)} ${waveId}`;
  const existing = state.waveforms.find((item) => item.name === waveId);
  const row = { id: existing?.id ?? `w${state.ids.waveform++}`, name: waveId, command: waveform };
  if (existing) Object.assign(existing, row);
  else state.waveforms.push(row);
  renderAll();
}

function waveformNames() {
  return state.waveforms.map((item) => item.name);
}

function addSource() {
  const waveId = value("sourceWaveformId");
  if (!waveId) return setStatus("Add or choose a waveform before adding a source.");
  const sourceType = value("sourceType");
  const extra = value("sourceExtra");
  const position = [
    fmt(convertLength(num("sourceX"), value("sourceXUnit"))),
    fmt(convertLength(num("sourceY"), value("sourceYUnit"))),
    fmt(convertLength(num("sourceZ"), value("sourceZUnit"))),
  ].join(" ");
  const source = sourceType === "voltage_source" || sourceType === "transmission_line"
    ? `#${sourceType}: ${value("sourcePol") || "z"} ${position} ${extra || "50"} ${waveId}`
    : `#${sourceType}: ${value("sourcePol") || "z"} ${position} ${waveId}${extra ? ` ${extra}` : ""}`;
  state.sources.push({ id: `s${state.ids.source++}`, command: source });
  renderAll();
}

function addRx() {
  const x = fmt(convertLength(num("rxX"), value("rxXUnit")));
  const y = fmt(convertLength(num("rxY"), value("rxYUnit")));
  const z = fmt(convertLength(num("rxZ"), value("rxZUnit")));
  const id = value("rxId");
  state.receivers.push({ id: `r${state.ids.rx++}`, command: `#rx: ${x} ${y} ${z}${id ? ` ${id}` : ""}` });
  renderAll();
}

function setSnapshotDomainFromModel() {
  const d = domainValues();
  const maxFields = [
    ["snapXmax", d.x],
    ["snapYmax", d.y],
    ["snapZmax", d.z],
  ];
  for (const id of ["snapXmin", "snapYmin", "snapZmin"]) $(id).value = "0";
  for (const [id, metres] of maxFields) {
    $(id).value = fmt(fromBaseLength(metres, value(`${id}Unit`)));
  }
}

function setSnapshotTimeDefaults() {
  const tw = convertTime(num("timeWindow"), value("timeUnit"));
  const unit = value("snapshotTimeUnit");
  const start = Math.max(tw / 10, convertTime(1e-6, unit));
  $("snapshotStart").value = fmt(fromBaseTime(start, unit));
  $("snapshotStop").value = fmt(fromBaseTime(tw, unit));
  $("snapshotThird").value = value("snapshotRangeMode") === "arange" ? fmt(fromBaseTime((tw - start) / 10, unit)) : "11";
}

function updateSnapshotRangeMode() {
  const mode = value("snapshotRangeMode");
  $("snapshotStartWrap").firstChild.nodeValue = mode === "single" ? "Time" : "Start";
  $("snapshotStopWrap").style.display = mode === "single" ? "none" : "grid";
  $("snapshotThirdWrap").style.display = mode === "single" ? "none" : "grid";
  $("snapshotThirdWrap").firstChild.nodeValue = mode === "arange" ? "Step" : "Points";
}

function snapshotTimes() {
  const unit = value("snapshotTimeUnit");
  const positiveMin = convertTime(1e-12, unit);
  const start = Math.max(convertTime(num("snapshotStart"), unit), positiveMin);
  const mode = value("snapshotRangeMode");
  if (mode === "single") return [start];
  const stop = convertTime(num("snapshotStop"), unit);
  if (mode === "linspace") {
    const count = Math.max(1, Math.round(num("snapshotThird")));
    if (count === 1) return [start];
    return Array.from({ length: count }, (_, i) => start + ((stop - start) * i) / (count - 1));
  }
  const step = convertTime(num("snapshotThird"), unit);
  const end = stop;
  if (step <= 0) return [start];
  const count = Math.max(1, Math.floor((end - start) / step + 1e-9) + 1);
  return Array.from({ length: count }, (_, i) => start + step * i).filter((time) => time <= end + Math.abs(step) * 1e-9);
}

function validSnapshotTimes() {
  const times = snapshotTimes().filter((time) => time > 0);
  return times.length ? times : [convertTime(1e-12, value("snapshotTimeUnit"))];
}

function snapshotFilename(base, index, count) {
  const name = base || "snapshot";
  if (count === 1) return name;
  return `${name}_${String(index + 1).padStart(3, "0")}`;
}

function addSnapshot() {
  const fields = ["snapXmin", "snapYmin", "snapZmin", "snapXmax", "snapYmax", "snapZmax", "snapDx", "snapDy", "snapDz"];
  const converted = fields.map((id) => fmt(convertLength(num(id), value(`${id}Unit`))));
  const times = validSnapshotTimes();
  const baseName = value("snapshotName") || "snapshot";
  const commands = times.map((time, index) => `#snapshot: ${converted.join(" ")} ${fmt(time)} ${snapshotFilename(baseName, index, times.length)}`);
  state.snapshots.push({ id: `sn${state.ids.snapshot++}`, command: commands.join("\n") });
  renderAll();
}

function commandRows(containerId, items, remove) {
  const container = $(containerId);
  container.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "note";
    empty.textContent = "No commands added.";
    container.appendChild(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "command-row";
    const code = document.createElement("code");
    code.textContent = item.command;
    const button = document.createElement("button");
    button.className = "remove";
    button.textContent = "Remove";
    button.addEventListener("click", () => remove(item.id));
    row.append(code, button);
    container.appendChild(row);
  }
}

function displayCommandBlock(containerId, command) {
  const container = $(containerId);
  container.replaceChildren();
  const row = document.createElement("div");
  row.className = "command-row";
  row.style.gridTemplateColumns = "1fr";
  const code = document.createElement("code");
  code.textContent = command;
  row.appendChild(code);
  container.appendChild(row);
}

function renderCommands() {
  displayCommandBlock("modelCommandRows", modelCommands());
  commandRows("materialCommandRows", [...state.materials, ...state.poles.map((p) => ({ ...p, command: poleCommand(p) }))], (id) => {
    if (String(id).startsWith("m")) removeMaterial(id);
    else {
      state.poles = state.poles.filter((p) => p.id !== id);
      renderAll();
    }
  });
  commandRows("geometryCommandRows", state.geometry, (id) => {
    state.geometry = state.geometry.filter((g) => g.id !== id);
    renderAll();
  });
  commandRows("waveformCommandRows", state.waveforms, (id) => {
    state.waveforms = state.waveforms.filter((x) => x.id !== id);
    renderAll();
  });
  commandRows("sourceCommandRows", state.sources, (id) => {
    state.sources = state.sources.filter((x) => x.id !== id);
    renderAll();
  });
  commandRows("rxCommandRows", state.receivers, (id) => {
    state.receivers = state.receivers.filter((x) => x.id !== id);
    renderAll();
  });
  commandRows("snapshotCommandRows", state.snapshots, (id) => {
    state.snapshots = state.snapshots.filter((x) => x.id !== id);
    renderAll();
  });
  const parts = [
    modelCommands(),
    state.materials.map((m) => m.command).join("\n"),
    state.poles.map(poleCommand).join("\n"),
    state.geometry.map((g) => g.command).join("\n"),
    state.waveforms.map((s) => s.command).join("\n"),
    state.sources.map((s) => s.command).join("\n"),
    state.receivers.map((r) => r.command).join("\n"),
    state.snapshots.map((s) => s.command).join("\n"),
    state.other.map((item) => item.command).join("\n"),
  ].filter((part) => part.trim());
  $("outputCommands").value = `${parts.join("\n\n")}\n`;
}

function syncSelects() {
  for (const id of ["dispMaterial", "shapeMaterial", "shapeMaterialX", "shapeMaterialY", "shapeMaterialZ"]) {
    const selected = value(id);
    $(id).replaceChildren(...materialNames().map((name) => new Option(name, name)));
    if (materialNames().includes(selected)) $(id).value = selected;
  }
  const selectedWaveform = value("sourceWaveformId");
  $("sourceWaveformId").replaceChildren(...waveformNames().map((name) => new Option(name, name)));
  if (waveformNames().includes(selectedWaveform)) $("sourceWaveformId").value = selectedWaveform;
}

function renderSuggestions() {
  $("dxSuggestions").textContent = dxSuggestionSet().text;
  $("timeSuggestions").textContent = timeSuggestionSet().text;
}

function renderSnapshotInfo() {
  const d = domainValues();
  const tw = convertTime(num("timeWindow"), value("timeUnit"));
  const snapUnit = value("snapshotTimeUnit");
  const times = validSnapshotTimes();
  const last = times.length ? times[times.length - 1] : 0;
  $("snapshotInfo").textContent = [
    `Current domain: x 0 to ${fmt(d.x)} m, y 0 to ${fmt(d.y)} m, z 0 to ${fmt(d.z)} m`,
    `Time window: 0 to ${fmt(tw)} s (${fmt(fromBaseTime(tw, snapUnit))} ${snapUnit})`,
    `Snapshot range: ${times.length} point(s), ${fmt(fromBaseTime(times[0] || 0, snapUnit))} to ${fmt(fromBaseTime(last, snapUnit))} ${snapUnit}`,
  ].join("\n");
}

function renderAll() {
  syncSelects();
  renderGeometryFields();
  updateDispersionText();
  updateSnapshotRangeMode();
  renderSuggestions();
  renderSnapshotInfo();
  renderCommands();
  setStatus("Generated text updated.");
}

function setStatus(message) {
  $("status").textContent = message;
}

function clearCommandState() {
  state.materials = [];
  state.poles = [];
  state.geometry = [];
  state.waveforms = [];
  state.sources = [];
  state.receivers = [];
  state.snapshots = [];
  state.other = [];
  state.ids = { material: 1, pole: 1, geometry: 1, waveform: 1, source: 1, rx: 1, snapshot: 1 };
}

function commandName(command) {
  const match = command.match(/^#([^:\s]+)\s*:/);
  return match ? match[1] : "";
}

function commandParts(command) {
  const index = command.indexOf(":");
  if (index === -1) return [];
  return command.slice(index + 1).trim().split(/\s+/).filter(Boolean);
}

function setSelectValue(id, wanted) {
  const select = $(id);
  const valueToSet = [...select.options].some((option) => option.value === wanted) ? wanted : select.options[0]?.value;
  if (valueToSet !== undefined) select.value = valueToSet;
}

function setModelFieldsFromCommand(name, parts) {
  if (name === "title") {
    $("titleValue").value = parts.join(" ") || "model";
    return true;
  }
  if (name === "domain" && parts.length >= 3) {
    $("domainXUnit").value = "m";
    $("domainYUnit").value = "m";
    $("domainZUnit").value = "m";
    $("domainX").value = parts[0];
    $("domainY").value = parts[1];
    $("domainZ").value = parts[2];
    return true;
  }
  if (name === "dx_dy_dz" && parts.length >= 3) {
    setSelectValue("dxChoice", "User defined");
    $("dxUnit").value = "m";
    $("dxX").value = parts[0];
    $("dxY").value = parts[1];
    $("dxZ").value = parts[2];
    return true;
  }
  if (name === "time_window" && parts.length >= 1) {
    setSelectValue("timeChoice", "User defined");
    $("timeUnit").value = "s";
    $("timeWindow").value = parts[0];
    return true;
  }
  return false;
}

function importCommand(command) {
  const name = commandName(command);
  const parts = commandParts(command);
  if (!name) return false;
  if (setModelFieldsFromCommand(name, parts)) return true;
  if (name === "material" && parts.length >= 5) {
    state.materials.push({ id: `m${state.ids.material++}`, name: parts[4], command });
    return true;
  }
  if (name.startsWith("add_dispersion_")) {
    state.poles.push({ id: `p${state.ids.pole++}`, command });
    return true;
  }
  if (Object.prototype.hasOwnProperty.call(geometrySchemas, name)) {
    const material = parts.find((part) => materialNames().includes(part)) || "";
    state.geometry.push({ id: `g${state.ids.geometry++}`, type: name, material, command });
    return true;
  }
  if (name === "waveform") {
    state.waveforms.push({ id: `w${state.ids.waveform++}`, name: parts[3] || `waveform${state.ids.waveform}`, command });
    return true;
  }
  if (["hertzian_dipole", "magnetic_dipole", "voltage_source", "transmission_line"].includes(name)) {
    state.sources.push({ id: `s${state.ids.source++}`, command });
    return true;
  }
  if (name === "rx") {
    state.receivers.push({ id: `r${state.ids.rx++}`, command });
    return true;
  }
  if (name === "snapshot") {
    state.snapshots.push({ id: `sn${state.ids.snapshot++}`, command });
    return true;
  }
  state.other.push({ command });
  return true;
}

function loadInputText(text, filename) {
  clearCommandState();
  const commands = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.startsWith("#") && line.includes(":"));
  let imported = 0;
  for (const command of commands) {
    if (importCommand(command)) imported += 1;
  }
  if (!state.materials.length) state.materials.push({ id: `m${state.ids.material++}`, name: "soil", command: "#material: 6 0.001 1 0 soil" });
  renderAll();
  const extras = state.other.length ? ` ${state.other.length} unclassified command(s) preserved in output.` : "";
  setStatus(`Loaded ${imported} command(s) from ${filename || ".in file"}.${extras}`);
}

function loadInputFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".in")) {
    setStatus("Choose a gprMax .in file.");
    return;
  }
  const reader = new FileReader();
  reader.addEventListener("load", () => loadInputText(String(reader.result || ""), file.name));
  reader.addEventListener("error", () => setStatus("Could not read the selected .in file."));
  reader.readAsText(file);
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".page").forEach((page) => page.classList.remove("active"));
      tab.classList.add("active");
      $(`page-${tab.dataset.tab}`).classList.add("active");
    });
  });
  $("addMaterial").addEventListener("click", addMaterial);
  $("addAnisotropic").addEventListener("click", addAnisotropicMaterials);
  $("addPole").addEventListener("click", addPole);
  $("addGeometry").addEventListener("click", addGeometry);
  $("addWaveform").addEventListener("click", addWaveform);
  $("addSource").addEventListener("click", addSource);
  $("addRx").addEventListener("click", addRx);
  $("addSnapshot").addEventListener("click", addSnapshot);
  $("fillSnapshotDomain").addEventListener("click", () => {
    setSnapshotDomainFromModel();
    renderAll();
  });
  $("shapeType").addEventListener("change", renderGeometryFields);
  $("dispType").addEventListener("change", updateDispersionText);
  $("snapshotRangeMode").addEventListener("change", () => {
    updateSnapshotRangeMode();
    setSnapshotTimeDefaults();
    renderAll();
  });
  $("dxChoice").addEventListener("change", () => {
    applyDxChoice();
    renderAll();
  });
  $("timeChoice").addEventListener("change", () => {
    applyTimeChoice();
    setSnapshotTimeDefaults();
    renderAll();
  });
  $("updateAll").addEventListener("click", () => {
    applyDxChoice();
    applyTimeChoice();
    setSnapshotTimeDefaults();
    renderAll();
  });
  $("loadInputFile").addEventListener("click", () => $("inputFilePicker").click());
  $("inputFilePicker").addEventListener("change", (event) => {
    loadInputFile(event.target.files[0]);
    event.target.value = "";
  });
  $("clearAll").addEventListener("click", () => {
    if (!window.confirm("Clear all material, geometry, source, receiver, and snapshot commands?")) return;
    clearCommandState();
    renderAll();
  });
  $("copyOutput").addEventListener("click", () => {
    navigator.clipboard.writeText($("outputCommands").value).then(() => setStatus("Copied generated .in text."));
  });
  document.addEventListener("input", (event) => {
    if (event.target.matches("input, select, textarea")) {
      if (["timeWindow", "timeUnit", "snapshotTimeUnit"].includes(event.target.id)) setSnapshotTimeDefaults();
      renderSuggestions();
      updateSnapshotRangeMode();
      renderSnapshotInfo();
      renderCommands();
    }
  });
}

function initialise() {
  for (const name of Object.keys(geometrySchemas)) $("shapeType").add(new Option(name, name));
  state.materials.push({ id: `m${state.ids.material++}`, name: "soil", command: "#material: 6 0.001 1 0 soil" });
  applyDxChoice();
  applyTimeChoice();
  setSnapshotDomainFromModel();
  setSnapshotTimeDefaults();
  const dx = dxSuggestionSet().values[dxSuggestionSet().recommended];
  $("snapDx").value = fmt(fromBaseLength(dx, value("snapDxUnit")));
  $("snapDy").value = fmt(fromBaseLength(dx, value("snapDyUnit")));
  $("snapDz").value = fmt(fromBaseLength(dx, value("snapDzUnit")));
  addWaveform();
  renderAll();
}

bindEvents();
initialise();
