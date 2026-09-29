const $ = id => document.getElementById(id);
const MODES = [["composite","Full"],["background","Background"],["layer","Layer"],["overlay","Overlay"],["strip","Progressive"],["alpha","Alpha"]];
let payload=null,annotation=null,humanAnswer="",mode="composite",selectedLayer=null,geometry=true,saveTimer=null,panStart=null;
let viewTransform={scale:1,tx:0,ty:0};

async function api(path,options={}) {
  const response=await fetch(path,{headers:{"Content-Type":"application/json"},...options});
  if(!response.ok){let message=`HTTP ${response.status}`;try{message=(await response.json()).detail||message}catch{}throw new Error(message)}
  return response.json();
}
const imageUrl=path=>`/api/image/${String(path).replaceAll("\\","/")}`;
const pretty=value=>JSON.stringify(value,null,2);
function setMessage(text=""){ $("message").textContent=text }
function setSave(text=""){ $("saveState").textContent=text }

function renderProgress(progress){
  $("reviewedCount").textContent=progress.reviewed;
  $("availableCount").textContent=progress.available;
  $("goalFill").style.width=`${progress.available?Math.min(100,progress.reviewed/progress.available*100):0}%`;
  $("counters").textContent=`✓ ${progress.accepted} · ✕ ${progress.rejected} · ⏸ ${progress.deferred} · ✎ ${progress.drafts} · ${progress.failed} failed`;
}

async function loadQuestion(questionId){
  setMessage("");setSave("Loading…");
  payload=await api(`/api/question/${encodeURIComponent(questionId)}`);
  annotation=structuredClone(payload.annotation);
  humanAnswer=payload.answer ?? payload.record?.answer ?? "";
  selectedLayer=annotation.gold_layer_ids?.[0] ?? payload.scene.layers?.[0]?.layer_id ?? null;
  mode="composite";
  renderAll();requestAnimationFrame(fitView);setSave("");
}

function renderAll(){
  const {scene,record}=payload;
  $("sceneId").textContent=scene.scene_id;
  $("questionCategory").textContent=record.category.replaceAll("_"," ");
  $("position").textContent=`#${payload.queue_position+1}/${payload.queue_total}`;
  const badge=$("statusBadge");badge.textContent=payload.status[0].toUpperCase()+payload.status.slice(1);badge.className=`status ${payload.status}`;
  renderProgress(payload.progress);
  $("questionText").textContent=record.question;
  $("modelStatus").textContent=record.generation?.status==="ok"?`model: ${record.validation?.status||"pass"}`:`generation: ${record.generation?.status||"missing"}`;
  $("confidence").textContent=`confidence: ${annotation.confidence||"—"}`;
  $("shardability").textContent=record.shardability||`${annotation.shards.length} shards`;
  renderModes();renderViewer();renderLayers();renderEditor();renderMetadata();renderActions();
}

function renderModes(){
  $("modeTabs").replaceChildren(...MODES.map(([key,label])=>{
    const button=document.createElement("button");button.textContent=label;button.className=mode===key?"active":"";
    button.onclick=()=>{mode=key;renderModes();renderViewer();requestAnimationFrame(fitView)};return button;
  }));
  $("geometryButton").textContent=geometry?"Geometry on":"Geometry off";
}
function stageImage(path,className=""){const img=document.createElement("img");img.src=imageUrl(path);img.className=className;return img}

function renderViewer(){
  const scene=payload.scene,stage=$("stage");stage.replaceChildren();stage.style.width=`${scene.width}px`;stage.style.height=`${scene.height}px`;applyViewTransform();
  const layer=scene.layers.find(item=>item.layer_id===selectedLayer);
  const selectedMissing=layer&&payload.file_checks[`layer_${layer.layer_id}`]===false;
  const missing=(mode==="composite"&&!payload.file_checks.full)||(mode==="background"&&!payload.file_checks.background)||(mode==="strip"&&!payload.file_checks.progressive_strip)||(["layer","alpha","overlay"].includes(mode)&&selectedMissing);
  if(missing){const node=document.createElement("div");node.className="missing";node.textContent="Image file missing";stage.append(node);return}
  let base=null;
  if(mode==="composite")base=stageImage(scene.full_path,"base");
  if(mode==="background")base=stageImage(scene.background?.path,"base");
  if(mode==="layer"&&layer)base=stageImage(layer.path,"base");
  if(mode==="overlay"&&layer){base=stageImage(scene.full_path,"base dim");stage.append(base,stageImage(layer.path))}
  else if(mode==="strip")base=stageImage(`${scene.render_dir}/progressive_strip.png`,"base");
  else if(mode==="alpha"&&layer){base=document.createElement("canvas");base.className="base";drawAlpha(base,layer.path)}
  if(base&&!stage.contains(base))stage.append(base);
  if(!base){const node=document.createElement("div");node.className="missing";node.textContent="Select a layer";stage.append(node)}
  if(geometry&&!['alpha','strip'].includes(mode))stage.append(geometryOverlay(scene));
}

function applyViewTransform(){const stage=$("stage");stage.style.transform=`translate(${viewTransform.tx}px,${viewTransform.ty}px) scale(${viewTransform.scale})`;if(payload)$("zoomInfo").textContent=`${payload.scene.width}×${payload.scene.height} · ${Math.round(viewTransform.scale*100)}%`}
function fitView(){if(!payload)return;const viewer=$("viewer"),scene=payload.scene,width=viewer.clientWidth,height=viewer.clientHeight;if(!width||!height)return;const scale=Math.min(width/scene.width,height/scene.height)*.97;viewTransform={scale,tx:(width-scene.width*scale)/2,ty:(height-scene.height*scale)/2};applyViewTransform()}
function actualSizeView(){if(!payload)return;const viewer=$("viewer"),scene=payload.scene;viewTransform={scale:1,tx:(viewer.clientWidth-scene.width)/2,ty:(viewer.clientHeight-scene.height)/2};applyViewTransform()}
function drawAlpha(canvas,path){const scene=payload.scene;canvas.width=scene.width;canvas.height=scene.height;const img=new Image();img.onload=()=>{const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);for(let i=0;i<pixels.data.length;i+=4){const a=pixels.data[i+3];pixels.data[i]=a;pixels.data[i+1]=a;pixels.data[i+2]=a;pixels.data[i+3]=255}ctx.putImageData(pixels,0,0)};img.src=imageUrl(path)}
function geometryOverlay(scene){
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox",`0 0 ${scene.width} ${scene.height}`);
  scene.layers.forEach((layer,index)=>{const box=layer.geometry?.bbox_xyxy;if(!box)return;const rect=document.createElementNS(svg.namespaceURI,"rect");rect.setAttribute("x",box[0]);rect.setAttribute("y",box[1]);rect.setAttribute("width",box[2]-box[0]);rect.setAttribute("height",box[3]-box[1]);rect.setAttribute("fill","none");rect.setAttribute("stroke",annotation.gold_layer_ids.includes(layer.layer_id)?"#16a34a":index%2?"#f97316":"#2563eb");rect.setAttribute("stroke-width",String(Math.max(1,scene.width/300)));rect.setAttribute("opacity",selectedLayer==null||selectedLayer===layer.layer_id?".98":".38");svg.append(rect)});return svg;
}

function renderLayers(){
  const scene=payload.scene,items=[];
  items.push(layerCard("Full image",scene.full_path,"composite",null,`${scene.width}×${scene.height}`));
  items.push(layerCard("Background · L0",scene.background.path,"background",0,scene.background.caption||"background"));
  scene.layers.forEach(layer=>items.push(layerCard(`Layer ${layer.layer_id}`,layer.path,"layer",layer.layer_id,layer.caption||layer.label||`z${layer.z_index}`)));
  const details=document.createElement("details");details.className="rail-details";const summary=document.createElement("summary");summary.textContent="Layer geometry details";const pre=document.createElement("pre");pre.textContent=pretty(scene.layers.map(layer=>({layer_id:layer.layer_id,label:layer.label,caption:layer.caption,geometry:layer.geometry,geometry_review:layer.geometry_review})));details.append(summary,pre);items.push(details);$("layerRail").replaceChildren(...items);
}
function layerCard(title,path,targetMode,layerId,caption){
  const selected=(layerId===selectedLayer&&["layer","alpha","overlay"].includes(mode))||(layerId===null&&mode===targetMode);
  const gold=layerId!==null&&annotation.gold_layer_ids.includes(layerId),distractor=layerId!==null&&annotation.distractor_layer_ids.includes(layerId);
  const card=document.createElement("div");card.className=`layer-card ${selected?"selected":""} ${gold?"gold":""} ${distractor?"distractor":""}`;
  const thumb=document.createElement("div");thumb.className="thumb";const img=document.createElement("img");img.src=imageUrl(path);thumb.append(img);
  const meta=document.createElement("div"),name=document.createElement("div");name.className="layer-name";name.textContent=title;if(layerId!==null){const chip=document.createElement("span");chip.className="chip";chip.textContent=gold?"GOLD":distractor?"DISTRACTOR":`L${layerId}`;name.append(chip)}
  const sub=document.createElement("div");sub.className="layer-caption";sub.textContent=caption;meta.append(name,sub);if(layerId!==null&&annotation.layer_roles[String(layerId)]){const role=document.createElement("div");role.className="role";role.textContent=annotation.layer_roles[String(layerId)];meta.append(role)}card.append(thumb,meta);
  card.onclick=()=>{selectedLayer=layerId;if(layerId===null)mode=targetMode;else if(!["layer","alpha","overlay"].includes(mode))mode="layer";renderModes();renderViewer();renderLayers();requestAnimationFrame(fitView)};return card;
}

function renderEditor(){
  const readOnly=Boolean(payload.decision);
  $("answerText").value=humanAnswer;$("answerText").disabled=readOnly;$("answerText").oninput=()=>{humanAnswer=$("answerText").value;changed()};
  $("shardGrid").replaceChildren(...annotation.shards.map((shard,index)=>{
    const card=document.createElement("div");card.className="shard-card";const head=document.createElement("div");head.className="shard-head";
    const number=document.createElement("span");number.className="shard-number";number.textContent=shard.shard_id;
    const role=document.createElement("input");role.className="role-input";role.value=shard.role;role.disabled=readOnly;role.title="Shard role";
    const units=document.createElement("input");units.className="unit-input";units.value=shard.source_unit_ids.join(", ");units.disabled=readOnly;units.title="Source unit IDs";
    const text=document.createElement("textarea");text.value=shard.text;text.disabled=readOnly;text.ariaLabel=`Shard ${shard.shard_id} text`;
    const update=()=>{shard.role=role.value.trim();shard.source_unit_ids=units.value.split(",").map(value=>Number(value.trim())).filter(Number.isInteger);shard.text=text.value;changed()};role.oninput=update;units.oninput=update;text.oninput=update;head.append(number,role,units);card.append(head,text);return card;
  }));
  bindJson("goldLayers","gold_layer_ids",readOnly);bindJson("minimalSets","minimal_sufficient_layer_sets",readOnly);bindJson("layerRoles","layer_roles",readOnly);bindJson("shardSupport","question_shard_support",readOnly);bindJson("dependencies","evidence_dependencies",readOnly);bindJson("distractors","distractor_layer_ids",readOnly);
  bindJson("atomicUnits","atomic_units",readOnly);
  $("evidenceNotes").value=annotation.evidence_notes||"";$("evidenceNotes").disabled=readOnly;$("evidenceNotes").oninput=()=>{annotation.evidence_notes=$("evidenceNotes").value;changed()};
  $("confidenceEdit").value=annotation.confidence;$("confidenceEdit").disabled=readOnly;$("confidenceEdit").onchange=()=>{annotation.confidence=$("confidenceEdit").value;$("confidence").textContent=`confidence: ${annotation.confidence}`;changed()};
}
function bindJson(id,key,readOnly){const field=$(id);field.value=pretty(annotation[key]);field.disabled=readOnly;field.oninput=()=>{try{annotation[key]=JSON.parse(field.value);field.classList.remove("invalid");changed(key)}catch{field.classList.add("invalid");setSave("Invalid JSON")}}}
function changed(key){if(key&&["gold_layer_ids","layer_roles","distractor_layer_ids"].includes(key)){renderLayers();renderViewer()}clearTimeout(saveTimer);setSave("Unsaved…");saveTimer=setTimeout(saveDraft,650)}

function renderMetadata(){
  $("validationInfo").textContent=pretty(payload.record.validation);
  $("generationInfo").textContent=pretty(payload.record.generation);
  $("sceneInfo").textContent=pretty({captions:payload.scene.captions,background:payload.scene.background,layers:payload.scene.layers,excluded_layers:payload.scene.excluded_layers,relations:payload.scene.relations,pairwise_geometry:payload.scene.pairwise_geometry,categories:payload.scene.categories,review:payload.scene.review,provenance:payload.scene.provenance});
  $("rawRecord").textContent=pretty(payload.record);
}
function renderActions(){const finalized=Boolean(payload.decision);$("acceptButton").disabled=finalized;$("rejectButton").disabled=finalized;$("deferButton").disabled=finalized}

async function saveDraft(){if(payload.decision)return;try{setSave("Saving…");const result=await api(`/api/question/${encodeURIComponent(payload.record.question_id)}/draft`,{method:"POST",body:JSON.stringify({annotation,answer:humanAnswer})});payload.progress=result.progress;payload.status="draft";renderProgress(result.progress);const badge=$("statusBadge");badge.textContent="Draft";badge.className="status draft";setSave("Saved ✓")}catch(error){setSave("Not saved");setMessage(error.message)}}
async function finalize(decision){clearTimeout(saveTimer);setMessage("");if(decision==="rejected"&&!confirm("Reject this generated annotation and continue?"))return;try{const result=await api(`/api/question/${encodeURIComponent(payload.record.question_id)}/finalize`,{method:"POST",body:JSON.stringify({decision,annotation,answer:humanAnswer})});if(result.next_question_id)await loadQuestion(result.next_question_id);else{payload.progress=result.progress;renderProgress(result.progress);setMessage("Review queue complete.")}}catch(error){setMessage(error.message)}}
async function deferCurrent(){clearTimeout(saveTimer);await saveDraft();try{const result=await api(`/api/question/${encodeURIComponent(payload.record.question_id)}/defer`,{method:"POST"});if(result.next_question_id)await loadQuestion(result.next_question_id)}catch(error){setMessage(error.message)}}
async function navigate(direction,questionId=null){try{clearTimeout(saveTimer);if(payload&&!payload.decision)await saveDraft();const result=await api("/api/navigate",{method:"POST",body:JSON.stringify({direction,question_id:questionId})});await loadQuestion(result.question_id)}catch(error){setMessage(error.message)}}

$("geometryButton").onclick=()=>{geometry=!geometry;renderModes();renderViewer()};$("fitButton").onclick=fitView;$("actualSizeButton").onclick=actualSizeView;
$("viewer").addEventListener("wheel",event=>{if(!payload)return;event.preventDefault();const rect=$("viewer").getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,scale=Math.min(40,Math.max(.05,viewTransform.scale*Math.pow(1.0018,-event.deltaY))),ratio=scale/viewTransform.scale;viewTransform={scale,tx:x-(x-viewTransform.tx)*ratio,ty:y-(y-viewTransform.ty)*ratio};applyViewTransform()},{passive:false});
$("viewer").addEventListener("pointerdown",event=>{if(event.button!==0)return;panStart={x:event.clientX,y:event.clientY,tx:viewTransform.tx,ty:viewTransform.ty};$("viewer").classList.add("dragging");$("viewer").setPointerCapture(event.pointerId)});
$("viewer").addEventListener("pointermove",event=>{if(!panStart)return;viewTransform={...viewTransform,tx:panStart.tx+event.clientX-panStart.x,ty:panStart.ty+event.clientY-panStart.y};applyViewTransform()});
const stopPan=()=>{panStart=null;$("viewer").classList.remove("dragging")};$("viewer").addEventListener("pointerup",stopPan);$("viewer").addEventListener("pointercancel",stopPan);new ResizeObserver(fitView).observe($("viewer"));
$("acceptButton").onclick=()=>finalize("accepted");$("rejectButton").onclick=()=>finalize("rejected");$("deferButton").onclick=deferCurrent;$("previousButton").onclick=()=>navigate("previous");$("nextButton").onclick=()=>navigate("next");$("flaggedButton").onclick=()=>navigate("flagged");$("deferredButton").onclick=()=>navigate("deferred");
$("jumpForm").onsubmit=event=>{event.preventDefault();const value=$("jumpInput").value.trim();if(value){navigate(null,value);$("jumpInput").value=""}};
$("exportButton").onclick=async()=>{try{await api("/api/export",{method:"POST"});const link=document.createElement("a");link.href="/api/export/download";link.download="question-shards-reviewed.json";link.click()}catch(error){setMessage(error.message)}};
(async()=>{try{const boot=await api("/api/bootstrap");await loadQuestion(boot.question_id)}catch(error){setMessage(error.message);setSave("Not ready")}})();
