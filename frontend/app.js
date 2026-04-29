// ===== API Base URL =====
// Use the actual page origin so it works regardless of how user accessed it
const API_BASE = window.location.origin;

// ===== Three.js Static Imports =====
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

console.log('[App] app.js loaded, Three.js version:', THREE.REVISION);

// ===== DOM Elements =====
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const statusCard = document.getElementById('statusCard');
const statusBadge = document.getElementById('statusBadge');
const progressFill = document.getElementById('progressFill');
const statusMessage = document.getElementById('statusMessage');
const modelActions = document.getElementById('modelActions');
const previewDownload = document.getElementById('previewDownload');
const finalDownload = document.getElementById('finalDownload');
const viewerPlaceholder = document.getElementById('viewerPlaceholder');
const viewerCanvas = document.getElementById('viewerCanvas');
const resetViewBtn = document.getElementById('resetView');
const toggleWireframeBtn = document.getElementById('toggleWireframe');

// ===== Three.js Globals =====
let scene, camera, renderer, controls, currentModel, isWireframe = false;
let animationId;

// ===== Upload Handling =====
console.log('[App] Attaching event listeners...');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    console.log('[App] File input change fired', e.target.files);
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    } else {
        console.log('[App] No files selected');
    }
    e.target.value = '';
});

function handleFile(file) {
    console.log('[App] handleFile called:', file.name, file.type, file.size);
    if (!file.type.match(/image\/(jpeg|png|webp)/)) {
        alert('Please upload a JPG, PNG, or WebP image.');
        return;
    }
    startGeneration(file);
}

// ===== Generation Flow =====
async function startGeneration(file) {
    console.log('[App] startGeneration starting...');
    statusCard.style.display = 'block';
    modelActions.style.display = 'none';
    updateStatus('Pending', 0, 'Uploading image...');
    clearViewer();
    
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        console.log('[App] Uploading...');
        const res = await fetch(`${API_BASE}/generate-3d`, {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
        const data = await res.json();
        console.log('[App] Upload response:', data);
        const jobId = data.job_id;
        
        updateStatus('Processing', 5, 'Starting generation...');
        pollJob(jobId);
        
    } catch (err) {
        console.error('[App] Upload error:', err);
        updateStatus('Failed', 0, err.message);
    }
}

function pollJob(jobId) {
    console.log('[App] Starting poll for job:', jobId);
    const poll = async () => {
        try {
            const res = await fetch(`${API_BASE}/jobs/${jobId}`);
            const data = await res.json();
            console.log('[App] Poll status:', data.status, data.progress_percent + '%');
            
            updateStatus(data.status, data.progress_percent, data.message);
            
            if (data.preview_url && !currentModel) {
                loadModel(data.preview_url, 'preview');
                modelActions.style.display = 'flex';
                previewDownload.href = data.preview_url;
                previewDownload.download = `preview_${jobId}.glb`;
            }
            
            if (data.final_url && (!currentModel || currentModel.userData.type !== 'final')) {
                loadModel(data.final_url, 'final');
                finalDownload.style.display = 'inline-flex';
                finalDownload.href = data.final_url;
                finalDownload.download = `final_${jobId}.glb`;
            }
            
            if (data.status === 'completed' || data.status === 'failed') {
                return;
            }
            
            setTimeout(poll, 1000);
        } catch (err) {
            console.error('[App] Poll error:', err);
            setTimeout(poll, 2000);
        }
    };
    poll();
}

function updateStatus(status, progress, message) {
    statusBadge.textContent = status;
    statusBadge.className = 'status-badge';
    
    const s = status.toLowerCase();
    if (s === 'processing_coarse' || s === 'refining') {
        statusBadge.classList.add('processing');
    } else if (s === 'coarse_ready') {
        statusBadge.classList.add('ready');
    } else if (s === 'completed') {
        statusBadge.classList.add('completed');
    } else if (s === 'failed') {
        statusBadge.classList.add('failed');
    }
    
    progressFill.style.width = `${progress}%`;
    statusMessage.textContent = message;
}

// ===== Three.js Viewer =====
async function initViewer() {
    console.log('[App] initViewer starting...');
    const container = document.getElementById('viewerContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x13131f);
    
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(3, 2, 3);
    
    renderer = new THREE.WebGLRenderer({ canvas: viewerCanvas, antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
    mainLight.position.set(5, 10, 7);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 1024;
    mainLight.shadow.mapSize.height = 1024;
    scene.add(mainLight);
    
    const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
    fillLight.position.set(-5, 0, -5);
    scene.add(fillLight);
    
    const rimLight = new THREE.DirectionalLight(0xffaa88, 0.3);
    rimLight.position.set(0, 5, -8);
    scene.add(rimLight);
    
    controls = new OrbitControls(camera, viewerCanvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1;
    controls.maxDistance = 10;
    controls.target.set(0, 0, 0);
    
    const gridHelper = new THREE.GridHelper(10, 20, 0x2d2d44, 0x1e1e2e);
    gridHelper.position.y = -1;
    scene.add(gridHelper);
    
    animate();
    window.addEventListener('resize', onResize);
    console.log('[App] initViewer complete');
}

function animate() {
    animationId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function onResize() {
    const container = document.getElementById('viewerContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (camera && renderer) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }
}

async function loadModel(url, type) {
    console.log('[App] loadModel:', url, type);
    if (!scene) await initViewer();
    
    const loader = new GLTFLoader();
    
    loader.load(
        url,
        (gltf) => {
            console.log('[App] Model loaded:', type);
            if (currentModel) scene.remove(currentModel);
            
            const model = gltf.scene;
            model.userData.type = type;
            
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = maxDim > 0 ? 2 / maxDim : 1;
            
            model.position.sub(center);
            model.scale.setScalar(scale);
            model.position.y += 0.5;
            
            model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            if (isWireframe) {
                model.traverse((child) => {
                    if (child.isMesh && child.material) {
                        child.material.wireframe = true;
                    }
                });
            }
            
            scene.add(model);
            currentModel = model;
            
            viewerPlaceholder.style.display = 'none';
            viewerCanvas.style.display = 'block';
            
            if (controls) {
                controls.autoRotate = true;
                setTimeout(() => { controls.autoRotate = false; }, 3000);
            }
        },
        undefined,
        (error) => {
            console.error('[App] Error loading model:', error);
        }
    );
}

function clearViewer() {
    if (currentModel) {
        scene.remove(currentModel);
        currentModel = null;
    }
    viewerPlaceholder.style.display = 'flex';
    viewerCanvas.style.display = 'none';
}

// ===== Viewer Controls =====
resetViewBtn.addEventListener('click', () => {
    if (!camera || !controls) return;
    camera.position.set(3, 2, 3);
    controls.target.set(0, 0, 0);
    controls.update();
});

toggleWireframeBtn.addEventListener('click', () => {
    isWireframe = !isWireframe;
    toggleWireframeBtn.style.color = isWireframe ? '#667eea' : '#a0a0b0';
    
    if (currentModel) {
        currentModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.wireframe = isWireframe;
            }
        });
    }
});

console.log('[App] All event listeners attached, ready for uploads');
