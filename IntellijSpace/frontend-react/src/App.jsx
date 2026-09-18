import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import './App.css';

const FALLBACK_API_BASE = '/api/v1';

function getApiBase() {
  const fromEnv = import.meta.env.VITE_API_BASE;
  const fromQuery = new URLSearchParams(window.location.search).get('api_base');
  const fromStorage = window.localStorage.getItem('MVGGT_API_BASE');
  return (fromQuery || fromStorage || fromEnv || FALLBACK_API_BASE).replace(/\/$/, '');
}

function getZhName(name) {
  const l = (name || '').toLowerCase();
  if (l.includes('chair')) return '休闲椅';
  if (l.includes('sofa')) return '沙发';
  if (l.includes('bed')) return '床';
  if (l.includes('table')) return '桌子';
  if (l.includes('desk')) return '书桌';
  if (l.includes('cabinet')) return '柜子';
  if (l.includes('lamp')) return '灯具';
  if (l.includes('shelf')) return '置物架';
  if (l.includes('stool')) return '凳子';
  return '家具素材';
}

function SceneModel({ url, yawDeg, scaleMult, transformEnabled, onAssetCount, assetNodeNames }) {
  const gltf = useGLTF(url);
  const root = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

  const assetNodes = useMemo(() => {
    const nameSet = new Set(assetNodeNames || []);
    const nodes = [];

    if (nameSet.size > 0) {
      root.traverse((obj) => {
        if (obj?.name && nameSet.has(obj.name)) {
          obj.matrixAutoUpdate = false;
          nodes.push(obj);
        }
      });
    }

    if (nodes.length === 0) {
      const worldNode = root.children?.find((c) => c.name === 'world') || root;
      for (const child of worldNode.children || []) {
        if (child.name && !/^geometry_\d+/.test(child.name)) {
          child.matrixAutoUpdate = false;
          nodes.push(child);
        }
      }
    }

    return nodes;
  }, [root, assetNodeNames]);

  const baseTransforms = useMemo(() => {
    return assetNodes.map((node) => {
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      node.matrix.decompose(position, quaternion, scale);
      return { node, position, quaternion, scale };
    });
  }, [assetNodes]);

  useEffect(() => { onAssetCount(assetNodes.length); }, [assetNodes.length, onAssetCount]);

  useLayoutEffect(() => {
    if (assetNodes.length === 0) return;

    if (!transformEnabled) {
      for (const base of baseTransforms) {
        base.node.matrix.compose(base.position, base.quaternion, base.scale);
        base.node.matrixWorldNeedsUpdate = true;
        base.node.updateMatrixWorld(true);
      }
      return;
    }

    const yaw = THREE.MathUtils.degToRad(yawDeg);
    const yawQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    for (const base of baseTransforms) {
      const q = base.quaternion.clone().multiply(yawQ);
      const s = base.scale.clone().multiplyScalar(scaleMult);
      base.node.matrix.compose(base.position, q, s);
      base.node.matrixWorldNeedsUpdate = true;
      base.node.updateMatrixWorld(true);
    }
  }, [assetNodes, baseTransforms, scaleMult, transformEnabled, yawDeg]);

  return <primitive object={root} />;
}

function Viewer({ modelUrl, yawDeg, scaleMult, transformEnabled, onAssetCount, assetNodeNames }) {
  if (!modelUrl) {
    return (
      <div className="viewer-empty">
        <div className="empty-icon">📦</div>
        <p>等待模型生成...</p>
      </div>
    );
  }
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Canvas camera={{ position: [3, 3, 5], fov: 50 }} style={{ width: '100%', height: '100%' }}>
        <color attach="background" args={['#f5f5f3']} />
        <ambientLight intensity={1.0} />
        <directionalLight intensity={1.2} position={[3, 4, 5]} />
        <Suspense fallback={null}>
          <SceneModel key={modelUrl} url={modelUrl} yawDeg={yawDeg} scaleMult={scaleMult} transformEnabled={transformEnabled} onAssetCount={onAssetCount} assetNodeNames={assetNodeNames} />
        </Suspense>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}

function Modal({ show, title, message, onClose, children }) {
  if (!show) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-body">
          <div className="modal-icon">⚠️</div>
          <div>
            <h3 className="modal-title">{title}</h3>
            <p className="modal-msg">{message}</p>
          </div>
        </div>
        <div className="modal-footer">
          {children || <button className="btn-primary" onClick={onClose}>我知道了</button>}
        </div>
      </div>
    </div>
  );
}

const STEPS = ['upload', 'parse', 'infer', 'build', 'done'];
const STEP_LABELS = { upload: '上传', parse: '解析', infer: '推理', build: '生成', done: '完成' };

function App() {
  const apiBase = useMemo(getApiBase, []);
  const pollRef = useRef(null);

  const [landed, setLanded] = useState(false);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  const [assets, setAssets] = useState([]);
  const [assetQ, setAssetQ] = useState('');
  const [selAsset, setSelAsset] = useState('');
  const [prompt, setPrompt] = useState('');
  const [imgFiles, setImgFiles] = useState([]);
  const [vidFile, setVidFile] = useState(null);
  const [imgPreviews, setImgPreviews] = useState([]);

  const [status, setStatus] = useState('空闲');
  const [step, setStep] = useState('idle');
  const [curTask, setCurTask] = useState('');
  const [lastTask, setLastTask] = useState('');
  const [variant, setVariant] = useState('result');
  const [modelUrl, setModelUrl] = useState('');
  const [variantCache, setVariantCache] = useState({ result: '', nomask: '', mask: '' });
  const [busy, setBusy] = useState(false);

  const [metaAction, setMetaAction] = useState('-');
  const [metaTarget, setMetaTarget] = useState('-');
  const [metaAsset, setMetaAsset] = useState('-');

  const [isReplace, setIsReplace] = useState(false);
  const [hasPlace, setHasPlace] = useState(false);
  const [pivot, setPivot] = useState([0, 0, 0]);
  const [assetNodeNames, setAssetNodeNames] = useState([]);
  const [nodeCount, setNodeCount] = useState(0);
  const [yaw, setYaw] = useState(0);
  const [scale, setScale] = useState(1.0);
  const [statusCollapsed, setStatusCollapsed] = useState(true);

  const [modal, setModal] = useState({ show: false, title: '', msg: '', extra: null });
  const closeModal = () => setModal((m) => ({ ...m, show: false, extra: null }));
  const variantCacheRef = useRef({ result: '', nomask: '', mask: '' });

  const sliderOn = isReplace && hasPlace && variant === 'result';

  const filtered = useMemo(() => {
    const q = assetQ.trim().toLowerCase();
    return q ? assets.filter((a) => (a.name || '').toLowerCase().includes(q)) : assets;
  }, [assetQ, assets]);

  function variantUrl(tid, v) { return `${apiBase}/tasks/${tid}/download/${v}?t=${Date.now()}`; }
  function revokeVariantCache(cache) {
    for (const key of ['result', 'nomask', 'mask']) {
      const u = cache[key];
      if (u && u.startsWith('blob:')) URL.revokeObjectURL(u);
    }
  }

  useEffect(() => {
    fetch(`${apiBase}/assets`).then((r) => r.json()).then(setAssets).catch(() => setStatus('素材加载失败'));
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      revokeVariantCache(variantCacheRef.current);
    };
  }, [apiBase]);

  useEffect(() => {
    const urls = imgFiles.map((f) => URL.createObjectURL(f));
    setImgPreviews(urls);
    return () => urls.forEach(URL.revokeObjectURL);
  }, [imgFiles]);

  const onAssetCount = useCallback((n) => setNodeCount(n), []);

  async function pickBackend() {
    try {
      const rt = await fetch(`${apiBase}/runtime/options`).then((r) => r.json());
      if (!rt.local_gpu_available) return 'hf_api';
      return new Promise((resolve) => {
        setModal({
          show: true, title: '选择推理方式',
          msg: `检测到本地GPU（${rt.gpu_name || 'GPU'}）。请选择本地推理或 Hugging Face API。`,
          extra: (
            <>
              <button className="btn-ghost" onClick={() => { closeModal(); resolve('hf_api'); }}>使用HF API</button>
              <button className="btn-primary" onClick={() => { closeModal(); resolve('local'); }}>使用本地GPU</button>
            </>
          ),
        });
      });
    } catch { return 'hf_api'; }
  }

  async function parsePromptApi(raw) {
    const fd = new FormData(); fd.append('raw_prompt', raw);
    const r = await fetch(`${apiBase}/prompt/parse`, { method: 'POST', body: fd });
    return r.json();
  }

  async function submit(example = false) {
    const raw = prompt.trim() || (example ? 'Replace the small white mini fridge placed in the corner of the room, next to a trash bin and a wooden cabinet.' : '');
    if (!raw) { setModal({ show: true, title: '输入缺失', msg: '请先输入设计指令。', extra: null }); return; }

    setBusy(true); setStep('upload'); setStatus(example ? '提交样例任务...' : '提交任务...');
    try {
      const parsed = await parsePromptApi(raw);
      if ((parsed.action || '').toUpperCase() === 'REPLACE' && !selAsset) {
        setModal({ show: true, title: '请先选择素材', msg: '该指令是替换操作，请先在右侧素材库选择一个替换素材。', extra: null });
        setStatus('等待选择素材'); setBusy(false); return;
      }
      const bm = await pickBackend();
      const fd = new FormData();
      fd.append('raw_prompt', raw); fd.append('interval', '1'); fd.append('backend_mode', bm);
      if (selAsset) fd.append('selected_asset_path', selAsset);
      let url = `${apiBase}/tasks`;
      if (example) { url = `${apiBase}/tasks/example`; fd.append('max_images', '5'); }
      else { imgFiles.forEach((f) => fd.append('images', f)); if (vidFile) fd.append('video', vidFile); }
      const resp = await fetch(url, { method: 'POST', body: fd });
      if (!resp.ok) throw new Error(resp.status);
      const t = await resp.json();
      setCurTask(t.task_id); setLastTask(t.task_id);
      poll(t.task_id);
    } catch (e) { setStatus(`提交失败: ${e.message}`); setStep('idle'); setBusy(false); }
  }

  function poll(tid) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${apiBase}/tasks/${tid}`);
        const st = await r.json();
        setStatus(`${(st.status || '').toUpperCase()} - ${st.message || ''}`);
        if (st.status === 'queued') setStep('parse');
        else if (st.status === 'running') setStep(st.message?.toLowerCase().includes('inference') ? 'infer' : 'parse');
        else if (st.status === 'success') { clearInterval(pollRef.current); pollRef.current = null; setStep('build'); await result(tid); setBusy(false); }
        else if (st.status === 'failed') { clearInterval(pollRef.current); pollRef.current = null; setStep('idle'); setBusy(false); }
      } catch { clearInterval(pollRef.current); pollRef.current = null; setStep('idle'); setBusy(false); }
    }, 1500);
  }

  async function result(tid) {
    const r = await fetch(`${apiBase}/tasks/${tid}/result`).then((x) => x.json());
    const rep = (r.action || '').toUpperCase() === 'REPLACE';
    const hp = !!r.has_placement;
    setMetaAction(r.action || '-');
    setMetaTarget(r.target_to_segment || '-');
    setMetaAsset(r.selected_asset_path?.split('/').pop() || '-');
    setIsReplace(rep); setHasPlace(hp);
    setPivot(Array.isArray(r.placement_pivot) && r.placement_pivot.length === 3 ? r.placement_pivot : [0, 0, 0]);
    setAssetNodeNames(Array.isArray(r.asset_node_names) ? r.asset_node_names : []);
    setYaw(0); setScale(1.0);
    const v = rep ? 'result' : 'nomask';
    const nextCache = { result: '', nomask: '', mask: '' };
    const variantList = ['result', 'nomask', 'mask'];
    await Promise.all(variantList.map(async (name) => {
      try {
        const resp = await fetch(variantUrl(tid, name));
        if (!resp.ok) return;
        const blob = await resp.blob();
        nextCache[name] = URL.createObjectURL(blob);
      } catch (_) {}
    }));
    revokeVariantCache(variantCacheRef.current);
    variantCacheRef.current = nextCache;
    setVariantCache(nextCache);
    setVariant(v);
    setModelUrl(nextCache[v] || variantUrl(tid, v));
    setStep('done'); setStatus('模型已生成');
  }

  function switchV(v) {
    if (!curTask) return;
    setVariant(v);
    setModelUrl(variantCache[v] || variantUrl(curTask, v));
  }

  async function download() {
    if (!lastTask) return;
    if (sliderOn) { await fetch(`${apiBase}/tasks/${lastTask}/dynamic?yaw_deg=${yaw}&scale=${scale}`).catch(() => {}); }
    window.open(`${apiBase}/tasks/${lastTask}/download`, '_blank');
  }

  function clearAll() {
    revokeVariantCache(variantCacheRef.current);
    const clearedCache = { result: '', nomask: '', mask: '' };
    variantCacheRef.current = clearedCache;
    setVariantCache(clearedCache);
    setPrompt(''); setImgFiles([]); setVidFile(null);
    setCurTask(''); setLastTask(''); setVariant('result'); setModelUrl('');
    setStatus('空闲'); setStep('idle');
    setIsReplace(false); setHasPlace(false); setPivot([0, 0, 0]); setAssetNodeNames([]);
    setYaw(0); setScale(1.0); setNodeCount(0);
    setMetaAction('-'); setMetaTarget('-'); setMetaAsset('-');
  }

  const stepIdx = STEPS.indexOf(step);
  const dotCls = step === 'done' ? 'dot green' : step === 'idle' ? 'dot gray' : 'dot orange pulse';

  if (!landed) {
    return (
      <div className="landing">
        <div className="landing-bg">
          <div className="blob b1" />
          <div className="blob b2" />
        </div>
        <div className="landing-content">
          <div className="landing-img-wrap">
            <img src="https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?q=80&w=2000&auto=format&fit=crop" alt="Interior" referrerPolicy="no-referrer" />
            <div className="landing-img-overlay" />
          </div>
          <div className="landing-text">
            <div className="landing-brand">
              <div className="brand-logo lg">智</div>
              <span className="brand-title lg">智维空间</span>
            </div>
            <h1 className="landing-h1">重塑你的<br /><em>居住美学</em></h1>
            <p className="landing-desc">让每一个关于家的想象，在这里精准落地。基于 AI 的室内设计云平台，开启你的灵感之旅。</p>
            <button className="landing-btn" onClick={() => setLanded(true)}>
              开启设计之旅 →
            </button>
            <div className="landing-stats">
              <div><strong>3D</strong><span>实时预览</span></div>
              <div className="divider" />
              <div><strong>AI</strong><span>智能设计</span></div>
              <div className="divider" />
              <div><strong>∞</strong><span>无限灵感</span></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-logo">智</div>
          <div>
            <div className="brand-title">智维空间 <span className="muted">IntelliSpace</span></div>
            <div className="brand-sub">让居住想象在这里被看见</div>
          </div>
        </div>
        <div className="topbar-r">
          <button className="pill">DIY绘制</button>
          <button className="pill">预览</button>
          <button className="pill dark">保存</button>
        </div>
      </header>

      <Modal show={modal.show} title={modal.title} message={modal.msg} onClose={closeModal}>
        {modal.extra}
      </Modal>

      <div className="main-area">
        {leftOpen && (
          <aside className="panel left">
            <div className="panel-head"><h2>🎨 输入与上传</h2></div>
            <div className="panel-body scroll">
              <label className="field-label">设计指令</label>
              <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="例如：把床换成单人床，删除书桌..." />

              <label className="field-label">图片素材 (多选)</label>
              <label className="dropzone">
                <span>📷</span><span>拖拽图片或点击上传</span>
                <input type="file" multiple accept="image/*" onChange={(e) => setImgFiles(Array.from(e.target.files || []))} />
              </label>

              <label className="field-label">视频素材 (单选)</label>
              <label className="dropzone">
                <span>🎬</span><span>拖拽视频或点击上传</span>
                <input type="file" accept="video/*" onChange={(e) => setVidFile(e.target.files?.[0] || null)} />
              </label>

              <button className="btn-primary w-full" disabled={busy} onClick={() => submit(false)}>执行设计</button>

              <div className="section-divider" />
              <label className="field-label">快速样例</label>
              <button className="example-card" disabled={busy} onClick={() => submit(true)}>
                <div className="example-top"><span>Corner Mini Fridge</span><span className="chip">一键运行</span></div>
                <p className="example-desc">Replace the small white mini fridge placed in the corner of the room, next to a trash bin and a wooden cabinet.</p>
        </button>

              {(imgPreviews.length > 0 || vidFile) && (
                <>
                  <div className="section-divider" />
                  <label className="field-label">上传预览</label>
                  <div className="preview-grid">
                    {imgPreviews.map((u, i) => <div key={i} className="preview-thumb"><img src={u} alt="" /></div>)}
                    {vidFile && <div className="preview-vid">🎬 {vidFile.name}</div>}
                  </div>
                </>
              )}
            </div>
          </aside>
        )}
        <button className="collapse-btn left-toggle" style={{ left: leftOpen ? 320 : 0 }} onClick={() => setLeftOpen((v) => !v)}>{leftOpen ? '‹' : '›'}</button>

        <main className="viewer-wrap">
          <div className="top-bar-float">
            <div className="stepper">
              {STEPS.map((s, i) => (
                <div key={s} className={`step-pill ${i <= stepIdx && step !== 'idle' ? 'active' : ''}`}>{STEP_LABELS[s]}</div>
              ))}
            </div>
            <div className="sep" />
            {curTask && (
              <div className="view-switch">
                {isReplace && <button className={`vbtn ${variant === 'result' ? 'on' : ''}`} onClick={() => switchV('result')}>结果</button>}
                <button className={`vbtn ${variant === 'nomask' ? 'on' : ''}`} onClick={() => switchV('nomask')}>无Mask</button>
                <button className={`vbtn ${variant === 'mask' ? 'on' : ''}`} onClick={() => switchV('mask')}>Mask</button>
              </div>
            )}
            <div className="tool-btns">
              <button className="icon-btn" title="清空" onClick={clearAll}>🗑</button>
              <button className="icon-btn" title="下载" onClick={download} disabled={!lastTask}>⬇</button>
            </div>
          </div>

          {sliderOn && (
            <div className="slider-panel">
              <div className="slider-head">
                <span>🎛 物体调节</span>
                <span className={nodeCount > 0 ? 'tag ok' : 'tag warn'}>{nodeCount > 0 ? `节点:${nodeCount}` : '搜索中'}</span>
              </div>
              <div className="slider-row">
                <div className="slider-labels"><span>水平旋转</span><span className="val-chip">{Math.round(yaw)}°</span></div>
                <input type="range" min="-180" max="180" step="1" value={yaw} onChange={(e) => setYaw(+e.target.value)} />
              </div>
              <div className="slider-row">
                <div className="slider-labels"><span>缩放比例</span><span className="val-chip">{scale.toFixed(2)}</span></div>
                <input type="range" min="0.1" max="3.0" step="0.01" value={scale} onChange={(e) => setScale(+e.target.value)} />
              </div>
        </div>
          )}

          <Viewer modelUrl={modelUrl} yawDeg={yaw} scaleMult={scale} transformEnabled={sliderOn} onAssetCount={onAssetCount} assetNodeNames={assetNodeNames} />

          <div className={`status-card ${statusCollapsed ? 'collapsed' : ''}`}>
            <div className="status-header">
              <div className="status-left"><div className={dotCls} /><span className="bold">任务状态</span></div>
              <div className="status-header-actions">
                <span className="step-label">{step.toUpperCase()}</span>
                <button
                  type="button"
                  className="status-toggle"
                  onClick={() => setStatusCollapsed((v) => !v)}
                  title={statusCollapsed ? '展开状态' : '折叠状态'}
                >
                  {statusCollapsed ? '展开' : '折叠'}
                </button>
              </div>
            </div>
            <p className="status-msg">{status}</p>
            {!statusCollapsed && (
              <>
                <div className="progress-bar">
                  {STEPS.map((s, i) => <div key={s} className={`bar-seg ${i <= stepIdx && step !== 'idle' ? 'filled' : ''}`} />)}
                </div>
                <div className="meta-section">
                  <div className="meta-row"><span>操作类型</span><span className="meta-val">{metaAction}</span></div>
                  <div className="meta-row"><span>目标对象</span><span className="meta-val">{metaTarget}</span></div>
                  <div className="meta-row"><span>替换素材</span><span className="meta-val truncate">{metaAsset}</span></div>
        </div>
              </>
            )}
          </div>
        </main>

        <button className="collapse-btn right-toggle" style={{ right: rightOpen ? 320 : 0 }} onClick={() => setRightOpen((v) => !v)}>{rightOpen ? '›' : '‹'}</button>
        {rightOpen && (
          <aside className="panel right">
            <div className="panel-head">
              <h2>📦 本地素材库</h2>
              <span className="chip">{filtered.length}</span>
            </div>
            <div className="search-wrap">
              <input className="search" value={assetQ} onChange={(e) => setAssetQ(e.target.value)} placeholder="搜索素材名称..." />
            </div>
            <div className="asset-grid scroll">
              {filtered.map((a) => {
                const zh = getZhName(a.name);
                return (
                  <div key={a.path} className={`asset-card ${selAsset === a.path ? 'selected' : ''}`} onClick={() => setSelAsset(a.path)} title={a.name}>
                    <div className="asset-thumb">{zh[0]}</div>
                    <h4 className="asset-name">{zh}</h4>
                  </div>
                );
              })}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
