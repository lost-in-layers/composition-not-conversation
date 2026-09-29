const CATEGORIES = [
  ["direct_ask", "Direct Ask", "Single object or attribute"],
  ["attribute_binding", "Attribute Binding", "Target identified by an anchor"],
  ["spatial", "Spatial", "Relative position between objects"],
  ["occlusion", "Occlusion", "Front, behind, or overlap"],
  ["counting", "Counting", "Complete countable set"],
  ["multi_hop_reference", "Multi-hop Reference", "Chain of two or more relations"],
];
const MODES = [["composite","Full image"],["background","Background"],["layer","Layer"],["alpha","Alpha"],["overlay","Overlay"],["strip","Strip"]];
let payload = null;
let categories = [];
let primary = null;
let selectedLayer = null;
let mode = "composite";
let geometry = true;
let saveTimer = null;
let viewTransform = {scale:1,tx:0,ty:0};
let panStart = null;
const $ = (id) => document.getElementById(id);

async function api(path, init={}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json"}, ...init});
  const body = await response.json().catch(()=>null);
  if (!response.ok) throw new Error(body?.detail || response.statusText);
  return body;
}
const imageUrl = (path) => `/api/image/${String(path||"").split("/").map(encodeURIComponent).join("/")}`;

function setMessage(text="") { $("message").textContent = text; }
function setSave(text="") { $("saveState").textContent = text; }

function renderProgress(progress) {
  $("questionCount").textContent = progress.question_slots;
  $("goalFill").style.width = `${Math.min(100, progress.question_slots / progress.question_target * 100)}%`;
  $("counters").textContent = `✓ ${progress.accepted} · ✕ ${progress.rejected} · ⏸ ${progress.deferred} · ✎ ${progress.drafts}`;
  $("categoryCounts").replaceChildren(...CATEGORIES.map(([key,label])=>{
    const chip=document.createElement("span");chip.textContent=`${label} ${progress.category_counts[key]||0}`;return chip;
  }));
}

function currentReview() {
  if (payload?.decision?.decision === "accepted") return payload.decision;
  return payload?.draft || {categories:[],primary_category:null};
}

async function loadScene(sceneId) {
  setMessage(""); setSave("Loading…");
  payload = await api(`/api/scene/${encodeURIComponent(sceneId)}`);
  const review = currentReview();
  categories = [...(review.categories || [])];
  primary = review.primary_category || null;
  selectedLayer = payload.scene.layers?.[0]?.layer_id ?? null;
  mode = "composite";
  renderAll();
  requestAnimationFrame(fitView);
  setSave("");
}

function renderAll() {
  const scene=payload.scene;
  $("sceneId").textContent=scene.scene_id;
  $("position").textContent=`#${payload.queue_position+1}/${payload.progress.available} available · ${payload.progress.selected_total} selected · ${scene.subset.toUpperCase()}`;
  const badge=$("statusBadge");badge.textContent=payload.status[0].toUpperCase()+payload.status.slice(1);badge.className=`status ${payload.status}`;
  renderProgress(payload.progress);
  renderModes();renderViewer();renderLayers();renderCategories();renderActions();
}

function renderModes() {
  $("modeTabs").replaceChildren(...MODES.map(([key,label])=>{
    const button=document.createElement("button");button.textContent=label;button.className=mode===key?"active":"";
    button.onclick=()=>{mode=key;renderModes();renderViewer();requestAnimationFrame(fitView);};return button;
  }));
  $("geometryButton").textContent=geometry?"Geometry on":"Geometry off";
}

function stageImage(path, className="") {
  const img=document.createElement("img");img.src=imageUrl(path);img.className=className;return img;
}

function renderViewer() {
  const scene=payload.scene,stage=$("stage");stage.replaceChildren();
  stage.style.width=`${scene.width}px`;
  stage.style.height=`${scene.height}px`;
  applyViewTransform();
  const layer=scene.layers.find(item=>item.layer_id===selectedLayer);
  const selectedMissing=layer&&payload.file_checks[`layer_${layer.layer_id}`]===false;
  const modeMissing=(mode==="composite"&&payload.file_checks.full===false)||(mode==="background"&&payload.file_checks.background===false)||(mode==="strip"&&payload.file_checks.progressive_strip===false)||(["layer","alpha"].includes(mode)&&selectedMissing);
  if(modeMissing){const missing=document.createElement("div");missing.className="missing";missing.textContent="Image file missing";stage.append(missing);return;}
  let base=null;
  if(mode==="composite") base=stageImage(scene.full_path,"base");
  if(mode==="background") base=stageImage(scene.background?.path,"base");
  if(mode==="layer"&&layer) base=stageImage(layer.path,"base");
  if(mode==="overlay"&&layer){base=stageImage(scene.full_path,"base dim");stage.append(base,stageImage(layer.path));}
  else if(mode==="strip") base=stageImage(`${scene.render_dir}/progressive_strip.png`,"base");
  else if(mode==="alpha"&&layer){base=document.createElement("canvas");base.className="base";drawAlpha(base,layer.path);}
  if(base&&!stage.contains(base)) stage.append(base);
  if(!base){const missing=document.createElement("div");missing.className="missing";missing.textContent="Select a layer";stage.append(missing);}
  if(geometry&&mode!=="alpha"&&mode!=="strip") stage.append(geometryOverlay(scene));
}

function applyViewTransform() {
  const stage=$("stage");
  stage.style.transform=`translate(${viewTransform.tx}px, ${viewTransform.ty}px) scale(${viewTransform.scale})`;
  if(payload)$("zoomInfo").textContent=`${payload.scene.width}×${payload.scene.height} · ${Math.round(viewTransform.scale*100)}%`;
}

function fitView() {
  if(!payload)return;
  const viewer=$("viewer"),scene=payload.scene;
  const width=viewer.clientWidth,height=viewer.clientHeight;
  if(!width||!height||!scene.width||!scene.height)return;
  const scale=Math.min(width/scene.width,height/scene.height)*.97;
  viewTransform={scale,tx:(width-scene.width*scale)/2,ty:(height-scene.height*scale)/2};
  applyViewTransform();
}

function actualSizeView() {
  if(!payload)return;
  const viewer=$("viewer"),scene=payload.scene;
  viewTransform={scale:1,tx:(viewer.clientWidth-scene.width)/2,ty:(viewer.clientHeight-scene.height)/2};
  applyViewTransform();
}

function drawAlpha(canvas,path) {
  const scene=payload.scene;canvas.width=scene.width;canvas.height=scene.height;
  const img=new Image();img.onload=()=>{const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0);const data=ctx.getImageData(0,0,canvas.width,canvas.height);for(let i=0;i<data.data.length;i+=4){const a=data.data[i+3];data.data[i]=a;data.data[i+1]=a;data.data[i+2]=a;data.data[i+3]=255;}ctx.putImageData(data,0,0);};img.src=imageUrl(path);
}

function geometryOverlay(scene) {
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox",`0 0 ${scene.width} ${scene.height}`);
  scene.layers.forEach((layer,index)=>{const box=layer.geometry?.bbox_xyxy;if(!box)return;const rect=document.createElementNS(svg.namespaceURI,"rect");rect.setAttribute("x",box[0]);rect.setAttribute("y",box[1]);rect.setAttribute("width",box[2]-box[0]);rect.setAttribute("height",box[3]-box[1]);rect.setAttribute("fill","none");rect.setAttribute("stroke",index%2?"#f97316":"#2563eb");rect.setAttribute("stroke-width",String(Math.max(1,scene.width/300)));rect.setAttribute("opacity",selectedLayer==null||selectedLayer===layer.layer_id?".95":".35");svg.append(rect);});
  return svg;
}

function renderLayers() {
  const scene=payload.scene,items=[];
  items.push(layerCard("Full image",scene.full_path,"composite",null,`${scene.width}×${scene.height}`));
  if(scene.background)items.push(layerCard("Background",scene.background.path,"background",0,scene.background.captions?.blip2||"z0"));
  scene.layers.forEach(layer=>items.push(layerCard(`Layer ${layer.layer_id}`,layer.path,"layer",layer.layer_id,layer.label||layer.captions?.blip2||`z${layer.z_index}`)));
  const details=document.createElement("details");details.className="metadata";const summary=document.createElement("summary");summary.textContent="Scene metadata";const pre=document.createElement("pre");pre.textContent=JSON.stringify({captions:scene.captions,relations:scene.relations,annotation:scene.annotation},null,2);details.append(summary,pre);items.push(details);
  $("layerRail").replaceChildren(...items);
}

function layerCard(title,path,targetMode,layerId,caption) {
  const card=document.createElement("div");card.className=`layer-card ${(layerId===selectedLayer&&(mode==="layer"||mode==="alpha"||mode==="overlay"))||(layerId==null&&mode===targetMode)?"selected":""}`;
  const thumb=document.createElement("div");thumb.className="thumb";const img=document.createElement("img");img.src=imageUrl(path);thumb.append(img);
  const meta=document.createElement("div");const name=document.createElement("div");name.className="layer-name";name.textContent=title;if(layerId!=null){const chip=document.createElement("span");chip.className="chip";chip.textContent=`L${layerId}`;name.append(chip);}const sub=document.createElement("div");sub.className="layer-caption";sub.textContent=caption;meta.append(name,sub);card.append(thumb,meta);
  card.onclick=()=>{selectedLayer=layerId;if(layerId==null)mode=targetMode;else if(!["layer","alpha","overlay"].includes(mode))mode="layer";renderModes();renderViewer();renderLayers();requestAnimationFrame(fitView);};return card;
}

function renderCategories() {
  const readOnly=Boolean(payload.decision);
  $("categoryGrid").replaceChildren(...CATEGORIES.map(([key,label,hint])=>{
    const card=document.createElement("div");card.className=`category-card ${categories.includes(key)?"selected":""} ${primary===key?"primary":""}`;
    const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.checked=categories.includes(key);checkbox.disabled=readOnly;checkbox.ariaLabel=`${label} suitable`;checkbox.onchange=()=>toggleCategory(key);
    const text=document.createElement("div");const name=document.createElement("label");name.textContent=label;name.onclick=()=>{if(!readOnly)toggleCategory(key)};const small=document.createElement("div");small.className="hint";small.textContent=hint;const suggestion=payload.suggestions.find(item=>item.category===key);if(suggestion?.suggested){const suggested=document.createElement("span");suggested.className="suggested";suggested.textContent=" · suggested";name.append(suggested);}text.append(name,small);
    const button=document.createElement("button");button.className=`primary-button ${primary===key?"active":""}`;button.textContent=primary===key?"Primary ✓":"Primary";button.disabled=readOnly;button.onclick=()=>setPrimary(key);
    card.append(checkbox,text,button);return card;
  }));
}

function toggleCategory(key) {
  categories=categories.includes(key)?categories.filter(item=>item!==key):[...categories,key];
  if(primary===key&&!categories.includes(key))primary=null;
  renderCategories();scheduleSave();
}
function setPrimary(key) {if(!categories.includes(key))categories.push(key);primary=key;renderCategories();scheduleSave();}

function renderActions() {
  const finalized=Boolean(payload.decision);
  $("acceptButton").disabled=finalized;
  $("rejectButton").disabled=finalized;
  $("deferButton").disabled=finalized;
}

function scheduleSave() {
  clearTimeout(saveTimer);setSave("Unsaved…");
  saveTimer=setTimeout(saveDraft,450);
}
async function saveDraft() {
  if(payload.decision)return;
  try{setSave("Saving…");const result=await api(`/api/scene/${encodeURIComponent(payload.scene.scene_id)}/draft`,{method:"POST",body:JSON.stringify({categories,primary_category:primary})});payload.progress=result.progress;payload.status=categories.length?"draft":"unseen";renderProgress(result.progress);$("statusBadge").textContent=payload.status==="draft"?"Draft":"Unseen";$("statusBadge").className=`status ${payload.status}`;setSave("Saved ✓");}
  catch(error){setSave("Save failed");setMessage(error.message);}
}

async function finalize(decision) {
  clearTimeout(saveTimer);setMessage("");
  if(decision==="rejected"&&!confirm("Reject this image and continue?"))return;
  try{const result=await api(`/api/scene/${encodeURIComponent(payload.scene.scene_id)}/finalize`,{method:"POST",body:JSON.stringify({decision,categories,primary_category:primary})});if(result.next_scene_id)await loadScene(result.next_scene_id);else{payload.progress=result.progress;renderProgress(result.progress);setMessage("Queue complete.");}}
  catch(error){setMessage(error.message);}
}

async function deferCurrent(){clearTimeout(saveTimer);await saveDraft();try{const result=await api(`/api/scene/${encodeURIComponent(payload.scene.scene_id)}/defer`,{method:"POST"});if(result.next_scene_id)await loadScene(result.next_scene_id);}catch(error){setMessage(error.message);}}
async function navigate(direction,sceneId=null){try{clearTimeout(saveTimer);if(payload&&!payload.decision)await saveDraft();const result=await api("/api/navigate",{method:"POST",body:JSON.stringify({direction,scene_id:sceneId})});await loadScene(result.scene_id);}catch(error){setMessage(error.message);}}

$("geometryButton").onclick=()=>{geometry=!geometry;renderModes();renderViewer();};
$("fitButton").onclick=fitView;
$("actualSizeButton").onclick=actualSizeView;
$("viewer").addEventListener("wheel",event=>{
  if(!payload)return;
  event.preventDefault();
  const rect=$("viewer").getBoundingClientRect();
  const x=event.clientX-rect.left,y=event.clientY-rect.top;
  const scale=Math.min(40,Math.max(.05,viewTransform.scale*Math.pow(1.0018,-event.deltaY)));
  const ratio=scale/viewTransform.scale;
  viewTransform={scale,tx:x-(x-viewTransform.tx)*ratio,ty:y-(y-viewTransform.ty)*ratio};
  applyViewTransform();
},{passive:false});
$("viewer").addEventListener("pointerdown",event=>{
  if(event.button!==0)return;
  panStart={x:event.clientX,y:event.clientY,tx:viewTransform.tx,ty:viewTransform.ty};
  $("viewer").classList.add("dragging");
  $("viewer").setPointerCapture(event.pointerId);
});
$("viewer").addEventListener("pointermove",event=>{
  if(!panStart)return;
  viewTransform={...viewTransform,tx:panStart.tx+event.clientX-panStart.x,ty:panStart.ty+event.clientY-panStart.y};
  applyViewTransform();
});
const stopPan=()=>{panStart=null;$("viewer").classList.remove("dragging");};
$("viewer").addEventListener("pointerup",stopPan);
$("viewer").addEventListener("pointercancel",stopPan);
new ResizeObserver(()=>fitView()).observe($("viewer"));
$("acceptButton").onclick=()=>finalize("accepted");$("rejectButton").onclick=()=>finalize("rejected");$("deferButton").onclick=deferCurrent;
$("previousButton").onclick=()=>navigate("previous");$("nextButton").onclick=()=>navigate("next");$("deferredButton").onclick=()=>navigate("deferred");
$("jumpForm").onsubmit=(event)=>{event.preventDefault();const value=$("jumpInput").value.trim();if(value){navigate(null,value);$("jumpInput").value="";}};
$("exportButton").onclick=async()=>{try{await api("/api/export",{method:"POST"});const link=document.createElement("a");link.href="/api/export/download";link.download="category_selections.json";link.click();}catch(error){setMessage(error.message);}};

(async()=>{try{const boot=await api("/api/bootstrap");await loadScene(boot.scene_id);}catch(error){setMessage(error.message);setSave("Not ready");}})();
