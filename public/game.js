// AI World Generator - 3D Interactive Game
// Uses Three.js for rendering and AI to generate environments

class WorldGenerator {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.objects = [];
    this.interactableObjects = [];
    this.lights = [];
    this.shadowBudget = 8; // safety cap to avoid exceeding GPU texture unit limits
    this.raycaster = new THREE.Raycaster();
    this.highlighted = null;
    this.interactDistance = 60;
    this.ground = null;

    // Reusable temp vectors to reduce per-frame allocations
    this._centerVec2 = new THREE.Vector2(0, 0);
    this._tmpVec3a = new THREE.Vector3();
    this._tmpVec3b = new THREE.Vector3();
    this.companion = {
      mesh: null,
      target: null,
      active: false,
      timer: null,
      message: '',
      speed: 40,
      model: 'gpt-oss-20',
      behavior: 'follow', // follow, explore, guard, waypoint
      lastDecision: 0,
      memory: [],
      personality: 'friendly', // friendly, curious, protective
      waypoint: null,
      waypointMarker: null,
      autoLootRange: 30,
      lastResourceCallout: 0,
      voiceEnabled: true,
      voiceRate: 1.02,
      voicePitch: 1.05,
      voiceVolume: 0.85
    };
    this.resident = {
      mesh: null,
      target: null,
      state: 'idle',
      hunger: 0,
      energy: 100,
      mood: 'curious',
      timer: 0
    };
    this.moveForward = false;
    this.moveBackward = false;
    this.moveLeft = false;
    this.moveRight = false;
    this.canJump = false;
    this.isRunning = false;
    this.velocity = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.prevTime = performance.now();
    this.playerHeight = 6;
    this.lastUIUpdate = 0;
    this.ui = {};
    this.inventory = [];
    this.maxInventory = 9;
    this.selectedSlot = 0;
    
    // Infinite world chunk system
    this.chunkSize = 80;
    this.renderDistance = 3;
    this.chunks = new Map();
    this.lastChunkUpdate = { x: 0, z: 0 };
    this.seed = Math.floor(Math.random() * 100000);
    
    // Weather and atmosphere
    this.particles = [];
    this.weatherType = 'clear'; // clear, rain, snow
    this.timeOfDay = 0.5; // 0=night, 0.5=noon, 1=night
    this.timeSpeed = 0.00005; // Day/night cycle speed
    
    // World memory persistence
    this.worldMemory = {
      playerPosition: { x: 0, y: 6, z: 30 },
      playerVelocity: { x: 0, y: 0, z: 0 },
      worldData: null,
      score: 0,
      playtime: 0,
      sessionStart: Date.now(),
      sessions: 0,
      inventory: []
    };
    this.autoSaveInterval = null;
    this.enhanced = null; // Will be initialized after scene is created
    this.sunLight = null;
    this.ambientLight = null;
    this.hemiLight = null;
    this.skyDome = null;
    this.skyUniforms = null;
    this.pmremGenerator = null;
    this.envMapTarget = null;
    this.envCanvas = null;
    this.envContext = null;
    this.envTexture = null;
    this.envUpdateInterval = 2000;
    this.lastEnvUpdate = 0;
    this._skyTopColor = new THREE.Color(0x8fb8ff);
    this._skyBottomColor = new THREE.Color(0xefe9dc);
    this._skyBaseColor = new THREE.Color(0x87ceeb);
    this._whiteColor = new THREE.Color(0xffffff);
    this._blackColor = new THREE.Color(0x000000);
    this._sunColor = new THREE.Color(0xffffff);
    this._ambientColor = new THREE.Color(0xb5d4ff);
    this.postFX = null;
    this.postFXEnabled = true;
    this.focusDistance = 45;
    
    this.init();
    this.setupEventListeners();
    this.cacheUIElements();
    this.loadWorldMemory();
    this.spawnResident();
    this.animate();
    
    // Load saved world or generate default
    if (this.worldMemory.worldData) {
      this.generateFromData(this.worldMemory.worldData);
    } else {
      this.generateDefaultWorld();
    }
  }

  init() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);
    this.scene.fog = new THREE.FogExp2(0x1a1a2e, 0.008);

    // Camera
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(0, this.playerHeight, 30);

    // Renderer - must be created before anything tries to render
    this.createRenderer();
    
    if (!this.renderer) {
      console.error('Failed to initialize renderer - world will not display');
      return;
    }
    this.setupGraphicsPipeline();

    // Initialize enhanced world system (dynamic lighting, particles, biomes)
    import('./vendor/EnhancedWorld.js').then(module => {
      const { EnhancedWorld } = module;
      this.enhanced = new EnhancedWorld(this);
      console.log('✓ Enhanced world system loaded');
    }).catch(err => {
      console.warn('Enhanced world system unavailable:', err);
    });

    // Pointer Lock Controls
    this.controls = new THREE.PointerLockControls(this.camera, document.body);
    this.scene.add(this.controls.getObject());

    // Click to lock
    document.addEventListener('click', () => {
      if (!this.controls.isLocked) {
        this.controls.lock();
      }
    });

    // Spawn resident after initial click to ensure context is ready
    document.addEventListener('click', () => {
      if (!this.resident.mesh) {
        this.spawnResident();
      }
    }, { once: true });

    // Handle resize
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  setStatus(text, className = '') {
    const statusEl = document.getElementById('status');
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = className;
  }

  safeGetEl(id) {
    const el = document.getElementById(id);
    if (!el) {
      console.warn(`Missing UI element #${id}`);
    }
    return el;
  }

  sanityCheckUI() {
    const required = ['generate-btn', 'prompt-input', 'status', 'controls-panel', 'panel-toggle'];
    const missing = required.filter(id => !document.getElementById(id));
    if (missing.length) {
      this.setStatus(`UI missing: ${missing.join(', ')}`, 'error');
    }
    return missing;
  }

  updateResidentUI(line, mood) {
    const lineEl = document.getElementById('resident-line');
    const moodEl = document.getElementById('resident-mood');
    if (lineEl && line) lineEl.textContent = line;
    if (moodEl && mood) moodEl.textContent = `Mood: ${mood}`;
  }

  addLight(light, { castShadow = false, parent = null } = {}) {
    // Keep shadow-casting lights under the GPU texture unit limit to avoid shader compile failures
    const maxUnits = this.renderer?.capabilities?.maxTextureImageUnits || 16;
    const maxShadowCasters = Math.max(1, Math.min(this.shadowBudget, maxUnits - 4));
    const currentShadowCasters = this.lights.filter(l => l.castShadow).length;
    const allowShadow = castShadow && currentShadowCasters < maxShadowCasters;

    light.castShadow = allowShadow;
    if (!allowShadow && castShadow) {
      console.warn('Shadow disabled to stay under texture unit budget', { currentShadowCasters, maxShadowCasters });
    }
    if (light.castShadow && light.shadow?.mapSize) {
      light.shadow.mapSize.set(1024, 1024);
    }

    (parent || this.scene).add(light);
    this.lights.push(light);
    return light;
  }

  createRenderer() {
    // Clean up any existing canvas
    const container = document.getElementById('game-container');
    if (!container) {
      console.error('game-container element not found!');
      return;
    }
    
    if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode === container) {
      container.removeChild(this.renderer.domElement);
      this.renderer.dispose();
    }

    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.configureRenderer();
      container.appendChild(this.renderer.domElement);
      
      console.log('✓ Renderer created successfully');

      // Handle WebGL context loss
      const canvas = this.renderer.domElement;
      canvas.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        this.setStatus('WebGL context lost. Click Reconnect.', 'error');
      });

      canvas.addEventListener('webglcontextrestored', () => {
        this.setStatus('Context restored. Rendering...', 'success');
        this.reconnectRenderer();
      });
    } catch (err) {
      console.error('Failed to create WebGL renderer:', err);
      this.setStatus('WebGL initialization failed: ' + err.message, 'error');
      // Try to display error to user
      container.innerHTML = `<div style="color: #f87171; padding: 20px; text-align: center;">
        <h2>WebGL Error</h2>
        <p>${err.message}</p>
        <p>Your browser may not support WebGL, or it may be disabled.</p>
      </div>`;
    }
  }

  configureRenderer() {
    if (!this.renderer) return;
    if (THREE.ColorManagement && 'enabled' in THREE.ColorManagement) {
      THREE.ColorManagement.enabled = true;
    }
    if ('outputColorSpace' in this.renderer && THREE.SRGBColorSpace) {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else if ('outputEncoding' in this.renderer && THREE.sRGBEncoding) {
      this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
    if ('toneMapping' in this.renderer && THREE.ACESFilmicToneMapping) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.05;
    }
    if ('physicallyCorrectLights' in this.renderer) {
      this.renderer.physicallyCorrectLights = true;
    }
  }

  setupGraphicsPipeline() {
    if (!this.renderer || !this.scene) return;
    if (this.pmremGenerator) {
      this.pmremGenerator.dispose();
      this.pmremGenerator = null;
    }
    if (this.envMapTarget) {
      this.envMapTarget.dispose();
      this.envMapTarget = null;
    }
    this.createSkyDome();
    this.updateEnvironmentMap(this._skyTopColor, this._skyBottomColor, true);
  }

  createSkyDome() {
    if (this.skyDome || !this.scene) return;
    const geometry = new THREE.SphereGeometry(520, 32, 15);
    this.skyUniforms = {
      topColor: { value: this._skyTopColor.clone() },
      bottomColor: { value: this._skyBottomColor.clone() },
      offset: { value: 33 },
      exponent: { value: 0.6 }
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this.skyUniforms,
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 bottomColor;
        uniform float offset;
        uniform float exponent;
        varying vec3 vWorldPosition;
        void main() {
          float h = normalize(vWorldPosition + vec3(0.0, offset, 0.0)).y;
          float mixVal = pow(max(h, 0.0), exponent);
          gl_FragColor = vec4(mix(bottomColor, topColor, mixVal), 1.0);
        }
      `,
      side: THREE.BackSide,
      depthWrite: false
    });
    this.skyDome = new THREE.Mesh(geometry, material);
    this.skyDome.renderOrder = -1;
    this.scene.add(this.skyDome);
  }

  updateEnvironmentMap(topColor, bottomColor, force = false) {
    if (!this.renderer || !this.scene) return;
    const now = performance.now();
    if (!force && now - this.lastEnvUpdate < this.envUpdateInterval) return;
    this.lastEnvUpdate = now;

    if (!this.envCanvas) {
      this.envCanvas = document.createElement('canvas');
      this.envCanvas.width = 512;
      this.envCanvas.height = 256;
      this.envContext = this.envCanvas.getContext('2d');
      this.envTexture = new THREE.CanvasTexture(this.envCanvas);
      this.envTexture.mapping = THREE.EquirectangularReflectionMapping;
      if ('colorSpace' in this.envTexture && THREE.SRGBColorSpace) {
        this.envTexture.colorSpace = THREE.SRGBColorSpace;
      } else if ('encoding' in this.envTexture && THREE.sRGBEncoding) {
        this.envTexture.encoding = THREE.sRGBEncoding;
      }
    }

    if (!this.envContext) return;
    const ctx = this.envContext;
    const gradient = ctx.createLinearGradient(0, 0, 0, this.envCanvas.height);
    gradient.addColorStop(0, `#${topColor.getHexString()}`);
    gradient.addColorStop(1, `#${bottomColor.getHexString()}`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, this.envCanvas.width, this.envCanvas.height);
    this.envTexture.needsUpdate = true;

    if (!this.pmremGenerator) {
      this.pmremGenerator = new THREE.PMREMGenerator(this.renderer);
      this.pmremGenerator.compileEquirectangularShader();
    }
    if (this.envMapTarget) {
      this.envMapTarget.dispose();
    }
    this.envMapTarget = this.pmremGenerator.fromEquirectangular(this.envTexture);
    this.scene.environment = this.envMapTarget.texture;
  }

  syncAtmosphere() {
    if (!this.scene || !this.skyUniforms) return;
    if (this.skyDome && this.camera) {
      this.skyDome.position.copy(this.camera.position);
    }
    const baseColor = this.scene.background instanceof THREE.Color
      ? this.scene.background
      : this._skyBaseColor;
    this._skyTopColor.copy(baseColor).lerp(this._whiteColor, 0.35);
    this._skyBottomColor.copy(baseColor).lerp(this._blackColor, 0.45);
    this.skyUniforms.topColor.value.copy(this._skyTopColor);
    this.skyUniforms.bottomColor.value.copy(this._skyBottomColor);
    this.updateEnvironmentMap(this._skyTopColor, this._skyBottomColor);
  }

  reconnectRenderer() {
    try {
      this.createRenderer();
      this.setupGraphicsPipeline();
      this.setStatus('Renderer reconnected.', 'success');
    } catch (err) {
      console.error('Reconnect failed:', err);
      this.setStatus('Reconnect failed: ' + err.message, 'error');
    }
  }

  setupEventListeners() {
    // Fail-soft if any key UI element is missing
    this.sanityCheckUI();

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
      switch (e.code) {
        case 'KeyW': this.moveForward = true; break;
        case 'KeyS': this.moveBackward = true; break;
        case 'KeyA': this.moveLeft = true; break;
        case 'KeyD': this.moveRight = true; break;
        case 'Space':
          if (this.canJump) {
            this.velocity.y = 220;
            this.canJump = false;
          }
          break;
        case 'ShiftLeft': this.isRunning = true; break;
        case 'KeyE': this.interactWithHighlighted(); break;
        case 'KeyF': this.addObjectFromUI(); break;
        case 'KeyQ': this.dropItem(); break;
        case 'Digit1': this.selectSlot(0); break;
        case 'Digit2': this.selectSlot(1); break;
        case 'Digit3': this.selectSlot(2); break;
        case 'Digit4': this.selectSlot(3); break;
        case 'Digit5': this.selectSlot(4); break;
        case 'Digit6': this.selectSlot(5); break;
        case 'Digit7': this.selectSlot(6); break;
        case 'Digit8': this.selectSlot(7); break;
        case 'Digit9': this.selectSlot(8); break;
      }
    });

    // Shift+Click for companion waypoint
    document.addEventListener('click', (e) => {
      if (e.shiftKey && this.companion.active) {
        const center = new THREE.Vector2(0, 0);
        this.raycaster.setFromCamera(center, this.camera);
        const targets = [...this.interactableObjects];
        if (this.ground) targets.push(this.ground);
        const hits = this.raycaster.intersectObjects(targets, true);
        if (hits.length > 0) {
          const point = hits[0].point;
          this.setCompanionWaypoint(point.x, point.z);
        }
      }
    });

    document.addEventListener('keyup', (e) => {
      switch (e.code) {
        case 'KeyW': this.moveForward = false; break;
        case 'KeyS': this.moveBackward = false; break;
        case 'KeyA': this.moveLeft = false; break;
        case 'KeyD': this.moveRight = false; break;
        case 'ShiftLeft': this.isRunning = false; break;
      }
    });

    // Generate button
    const generateBtn = this.safeGetEl('generate-btn');
    if (generateBtn) {
      generateBtn.addEventListener('click', () => this.generateFromAI());
    }

    // Model selector: gate browser-only WebLLM options when WebGPU is missing
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
      const browserOptions = Array.from(modelSelect.options).filter(opt => opt.value.startsWith('web-llm:'));
      const browserReady = () => navigator.gpu && typeof window !== 'undefined' && !!window.WebLLMBridge;
      if (!browserReady()) {
        browserOptions.forEach(opt => opt.disabled = true);
      }
      modelSelect.addEventListener('change', () => {
        if (modelSelect.value.startsWith('web-llm:') && !browserReady()) {
          alert('Browser LLM needs WebGPU and the WebLLM bridge. Switching to gpt-oss-20.');
          modelSelect.value = 'gpt-oss-20';
        }
      });
    }

    // Surface WebLLM download progress for in-browser generation
    window.addEventListener('webllm-progress', (evt) => {
      const detail = evt?.detail || {};
      const percent = Math.max(0, Math.round(detail.progress || 0));
      const text = detail.text || 'Preparing model...';
      const logEl = document.getElementById('llm-object-log');
      if (logEl && modelSelect && modelSelect.value.startsWith('web-llm:')) {
        logEl.textContent = `WebLLM ${percent}% - ${text}`;
      }
      const statusEl = document.getElementById('status');
      if (statusEl && statusEl.classList.contains('loading')) {
        statusEl.textContent = `WebLLM ${percent}% - ${text}`;
      }
    });

    // Memory controls
    const saveBtn = document.getElementById('save-world-btn');
    if (saveBtn) {
      saveBtn.addEventListener('click', () => {
        this.saveCurrentWorld();
        alert('🌍 World saved!');
      });
    }
    const reconnectBtn = document.getElementById('reconnect-btn');
    if (reconnectBtn) {
      reconnectBtn.addEventListener('click', () => {
        this.setStatus('Reconnecting renderer...', 'loading');
        this.reconnectRenderer();
      });
    }

    const closeInfoBtn = this.safeGetEl('close-info-btn');
    if (closeInfoBtn) {
      closeInfoBtn.addEventListener('click', () => {
        const panel = document.getElementById('info-panel');
        if (panel) {
          panel.style.display = 'none';
        }
      });
    }

    // Minimize controls panel
    const controlsPanel = this.safeGetEl('controls-panel');
    const togglePanelBtn = this.safeGetEl('panel-toggle');
    if (controlsPanel && togglePanelBtn) {
      const applyState = (isMinimized) => {
        controlsPanel.classList.toggle('minimized', isMinimized);
        togglePanelBtn.textContent = isMinimized ? '▶ Expand' : '➖ Minimize';
        togglePanelBtn.setAttribute('aria-expanded', (!isMinimized).toString());
        togglePanelBtn.setAttribute('aria-label', isMinimized ? 'Expand controls' : 'Minimize controls');
      };

      const saved = localStorage.getItem('panelMinimized') === '1';
      applyState(saved);

      togglePanelBtn.addEventListener('click', () => {
        const next = !controlsPanel.classList.contains('minimized');
        applyState(next);
        localStorage.setItem('panelMinimized', next ? '1' : '0');
      });
    }
    
    const clearBtn = document.getElementById('clear-memory-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        if (confirm('Clear all world memory? This cannot be undone.')) {
          this.clearWorldMemory();
          alert('🗑️ Memory cleared!');
        }
      });
    }

    // Object creation controls
    const addBtn = document.getElementById('add-object-btn');
    if (addBtn) {
      addBtn.addEventListener('click', () => this.addObjectFromUI());
    }

    // LLM-driven object creation
    const llmBtn = document.getElementById('llm-object-btn');
    const llmInput = document.getElementById('llm-object-input');
    const llmLog = document.getElementById('llm-object-log');
    if (llmBtn && llmInput) {
      llmBtn.addEventListener('click', () => {
        const prompt = llmInput.value.trim();
        if (!prompt) return;
        if (llmLog) llmLog.textContent = '🤖 Thinking...';
        this.generateObjectsFromLLM(prompt).then(msg => {
          if (llmLog) llmLog.textContent = msg;
        }).catch(err => {
          if (llmLog) llmLog.textContent = `Error: ${err.message}`;
        });
      });
    }

    // Companion controls
    const spawnBtn = document.getElementById('spawn-companion-btn');
    const pauseBtn = document.getElementById('pause-companion-btn');
    const companionModel = document.getElementById('companion-model');
    const companionPersonality = document.getElementById('companion-personality');
    const companionVoiceToggle = document.getElementById('companion-voice-toggle');
    if (companionModel) {
      this.companion.model = companionModel.value;
      companionModel.addEventListener('change', () => {
        this.companion.model = companionModel.value;
      });
    }
    if (companionPersonality) {
      this.companion.personality = companionPersonality.value;
      companionPersonality.addEventListener('change', () => {
        this.companion.personality = companionPersonality.value;
      });
    }
    if (companionVoiceToggle) {
      companionVoiceToggle.checked = this.companion.voiceEnabled;
      companionVoiceToggle.addEventListener('change', () => {
        this.companion.voiceEnabled = companionVoiceToggle.checked;
        if (!companionVoiceToggle.checked && window.speechSynthesis) {
          window.speechSynthesis.cancel();
        }
      });
    }
    if (spawnBtn) {
      spawnBtn.addEventListener('click', () => this.startCompanion());
    }
    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => this.stopCompanion());

        // Weather and time controls
        const weatherSelect = document.getElementById('weather-select');
        if (weatherSelect) {
          weatherSelect.addEventListener('change', () => {
            this.setWeather(weatherSelect.value);
            // Also update enhanced world if available
            if (this.enhanced) {
              this.enhanced.setWeather(weatherSelect.value);
            }
          });
        }
        const timeSpeed = document.getElementById('time-speed');
        if (timeSpeed) {
          timeSpeed.addEventListener('input', () => {
            this.timeSpeed = parseFloat(timeSpeed.value);
            // Also update enhanced world if available
            if (this.enhanced) {
              this.enhanced.setTimeSpeed(parseFloat(timeSpeed.value));
            }
          });
        }

        // Biome selection control
        const biomeSelect = document.getElementById('biome-select');
        if (biomeSelect) {
          biomeSelect.addEventListener('change', () => {
            if (this.enhanced) {
              this.enhanced.setBiome(biomeSelect.value);
            }
          });
        }
    }

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.dataset.preset;
        const presets = {
          forest: 'A dense magical forest with tall pine trees, glowing mushrooms scattered on the ground, moss-covered rocks, and fireflies floating in the air. The atmosphere is mystical with green and blue hues.',
          desert: 'A vast desert landscape with towering sand dunes, ancient pyramids in the distance, scattered oasis with palm trees, mysterious glowing crystals emerging from the sand, and a warm orange sunset.',
          snow: 'A frozen tundra with snow-covered pine trees, ice formations and glaciers, aurora borealis in the sky, frozen lake, snow-capped mountains in the background, and gentle snowfall.',
          alien: 'An alien planet with bizarre floating rock formations, bioluminescent plants in purple and cyan colors, strange crystalline structures, multiple moons in the sky, and an eerie atmosphere.',
          underwater: 'An underwater kingdom with colorful coral reefs, bioluminescent jellyfish, ancient sunken ruins covered in seaweed, schools of tropical fish, and rays of light filtering through the water.',
          city: 'A futuristic cyberpunk city with towering neon-lit skyscrapers, floating platforms, holographic billboards, flying vehicles, rain-slicked streets, and a purple-pink night sky.'
        };
        document.getElementById('prompt-input').value = presets[preset] || '';
      });
    });
  }

  cacheUIElements() {
    this.ui = {
      pos: document.getElementById('ui-pos'),
      speed: document.getElementById('ui-speed'),
      session: document.getElementById('ui-session'),
      playtime: document.getElementById('ui-playtime'),
      inventoryBar: document.getElementById('inventory-bar')
    };
  }

  selectSlot(index) {
    if (index >= 0 && index < this.maxInventory) {
      this.selectedSlot = index;
      this.updateInventoryUI();
    }
  }

  addToInventory(item) {
    if (this.inventory.length >= this.maxInventory) {
      this.setStatus('Inventory full!', 'error');
      return false;
    }
    this.inventory.push(item);
    this.worldMemory.inventory = [...this.inventory];
    this.updateInventoryUI();
    this.saveWorldMemory();
    return true;
  }

  formatItemLabel(type) {
    if (!type) return 'item';
    return type.charAt(0).toUpperCase() + type.slice(1);
  }

  removeWorldObject(obj) {
    if (!obj) return;
    this.scene.remove(obj);
    this.objects = this.objects.filter(o => o !== obj);
    this.interactableObjects = this.interactableObjects.filter(o => o !== obj);
    if (this.highlighted === obj) {
      this.highlightObject(obj, false);
      this.highlighted = null;
    }
  }

  deriveItemFromObject(obj, fallbackType = 'item') {
    if (!obj) return null;
    const type = obj.userData?.itemType || obj.userData?.type || fallbackType;
    const color = obj.userData?.itemColor || obj.material?.color?.getHex?.() || 0xffaa00;
    return { type: type || 'item', color };
  }

  pickupObject(obj, itemData) {
    const item = itemData || this.deriveItemFromObject(obj);
    if (!item) return false;
    if (this.addToInventory(item)) {
      this.removeWorldObject(obj);
      this.worldMemory.score += 10;
      this.displayMemoryStats();
      this.setStatus(`Picked up ${this.formatItemLabel(item.type)}`, 'success');
      return true;
    }
    return false;
  }

  removeFromInventory(index) {
    if (index >= 0 && index < this.inventory.length) {
      const item = this.inventory.splice(index, 1)[0];
      this.worldMemory.inventory = [...this.inventory];
      this.updateInventoryUI();
      this.saveWorldMemory();
      return item;
    }
    return null;
  }

  dropItem() {
    if (this.selectedSlot >= this.inventory.length) return;
    const item = this.removeFromInventory(this.selectedSlot);
    if (!item) return;
    
    const pos = this.camera.position.clone();
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    pos.add(dir.multiplyScalar(5));
    
    this.createItemInWorld(item, pos.x, pos.z);
  }

  createItemInWorld(item, x, z) {
    const geom = new THREE.BoxGeometry(1.5, 1.5, 1.5);
    const mat = new THREE.MeshStandardMaterial({ color: item.color || 0xffaa00, emissive: item.color || 0xffaa00, emissiveIntensity: 0.3 });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.type = 'item';
    mesh.userData.itemType = item.type;
    mesh.userData.itemColor = item.color;
    mesh.userData.pickupable = true;
    mesh.position.set(x, 1, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.objects.push(mesh);
    this.interactableObjects.push(mesh);
    return mesh;
  }

  updateInventoryUI() {
    if (!this.ui.inventoryBar) return;
    this.ui.inventoryBar.innerHTML = '';
    
    for (let i = 0; i < this.maxInventory; i++) {
      const slot = document.createElement('div');
      slot.className = 'inventory-slot';
      if (i === this.selectedSlot) slot.classList.add('selected');
      
      const num = document.createElement('div');
      num.className = 'slot-number';
      num.textContent = i + 1;
      slot.appendChild(num);
      
      if (i < this.inventory.length) {
        const item = this.inventory[i];
        slot.title = `${i + 1}: ${this.formatItemLabel(item.type)}`;
        const icon = document.createElement('div');
        icon.className = 'item-icon';
        icon.style.backgroundColor = item.color || '#ffaa00';
        icon.textContent = this.getItemEmoji(item.type);
        slot.appendChild(icon);
      }
      
      this.ui.inventoryBar.appendChild(slot);
    }
  }

  getItemEmoji(type) {
    const map = {
      crystal: '💎',
      tree: '🌲',
      rock: '🪨',
      mushroom: '🍄',
      floatingRock: '🪨',
      cube: '📦',
      sphere: '⚪',
      light: '💡',
      wood: '🪵'
    };
    return map[type] || '📦';
  }

  clearWorld() {
    // Remove all objects except controls
    this.objects.forEach(obj => this.scene.remove(obj));
    this.lights.forEach(light => this.scene.remove(light));
    this.objects = [];
    this.lights = [];
    this.interactableObjects = [];
    this.highlighted = null;
    this.ground = null;
    this.sunLight = null;
    this.ambientLight = null;
    this.hemiLight = null;
  }

  generateDefaultWorld() {
    this.clearWorld();
    
    // Setup global lighting
    const ambient = new THREE.AmbientLight(0xb5d4ff, 0.85);
    ambient.name = 'ambient';
    this.addLight(ambient);
    this.ambientLight = ambient;

    const hemi = new THREE.HemisphereLight(0x8fb8ff, 0x2a2416, 0.35);
    hemi.position.set(0, 120, 0);
    hemi.name = 'hemi';
    this.addLight(hemi);
    this.hemiLight = hemi;

    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.name = 'sun';
    sun.position.set(80, 120, 60);
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 10;
    sun.shadow.camera.far = 400;
    sun.shadow.camera.left = -200;
    sun.shadow.camera.right = 200;
    sun.shadow.camera.top = 200;
    sun.shadow.camera.bottom = -200;
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.05;
    sun.shadow.radius = 3;
    this.addLight(sun, { castShadow: true });
    this.sunLight = sun;

    this.scene.background = new THREE.Color(0x87ceeb);
    this.scene.fog = new THREE.FogExp2(0x87ceeb, 0.004);
    
    // Initialize chunks around player
    this.updateChunks();
  }
  
  hash(x, z) {
    // Simple hash for procedural generation
    let h = this.seed + x * 374761393 + z * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return (h ^ (h >> 16)) >>> 0;
  }
  
  random(x, z, offset = 0) {
    // Deterministic random based on position
    return (this.hash(x, z + offset) % 10000) / 10000;
  }
  
  getBiome(cx, cz) {
    // Determine biome based on chunk position with smoother transitions
    const noise1 = this.random(cx, cz, 1000);
    const noise2 = this.random(Math.floor(cx / 2), Math.floor(cz / 2), 2000);
    const combined = (noise1 + noise2) / 2;
    
    // Check neighboring chunks for biome blending
    const neighbors = [
      this.random(cx + 1, cz, 1000),
      this.random(cx - 1, cz, 1000),
      this.random(cx, cz + 1, 1000),
      this.random(cx, cz - 1, 1000)
    ];
    const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / neighbors.length;
    const blended = combined * 0.7 + avgNeighbor * 0.3;
    
    if (blended < 0.25) return 'forest';
    if (blended < 0.45) return 'plains';
    if (blended < 0.65) return 'village';
    if (blended < 0.85) return 'meadow';
    return 'mystical'; // rare magical biome
  }
  
  generateChunk(cx, cz) {
    const key = `${cx},${cz}`;
    if (this.chunks.has(key)) return;
    
    const chunkObjects = [];
    const baseX = cx * this.chunkSize;
    const baseZ = cz * this.chunkSize;
    const biome = this.getBiome(cx, cz);
    
    // Ground tile for this chunk
    const groundGeom = new THREE.PlaneGeometry(this.chunkSize, this.chunkSize, 10, 10);
    let groundColor, roughness;
    
    if (biome === 'forest') {
      groundColor = 0x2d5a27;
      roughness = 0.9;
    } else if (biome === 'village') {
      groundColor = 0x3a6b35;
      roughness = 0.8;
    } else if (biome === 'plains') {
      groundColor = 0x7cb342;
      roughness = 0.85;
    } else {
      groundColor = 0x8bc34a;
      roughness = 0.85;
    }
    
    const groundMat = new THREE.MeshStandardMaterial({ 
      color: groundColor,
      roughness: roughness,
      metalness: 0.1
    });
    
    // Add terrain variation
    const vertices = groundGeom.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      const wx = vertices[i] + baseX;
      const wz = vertices[i + 1] + baseZ;
      vertices[i + 2] = Math.sin(wx * 0.05) * Math.cos(wz * 0.05) * 4 + 
                       Math.sin(wx * 0.1) * Math.cos(wz * 0.1) * 2;
    }
    groundGeom.computeVertexNormals();
    
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(baseX, 0, baseZ);
    ground.receiveShadow = true;
    ground.userData.isGround = true;
    ground.userData.chunk = key;
    this.scene.add(ground);
    chunkObjects.push(ground);
    this.objects.push(ground);
    
    // Generate biome-specific content
    if (biome === 'forest') {
      // Dense forest with many trees
      const treeCount = 8 + Math.floor(this.random(cx, cz, 100) * 8);
      for (let i = 0; i < treeCount; i++) {
        const localX = this.random(cx, cz, i * 10) * this.chunkSize - this.chunkSize / 2;
        const localZ = this.random(cx, cz, i * 10 + 5000) * this.chunkSize - this.chunkSize / 2;
        const height = 10 + this.random(cx, cz, i * 10 + 8000) * 12;
        const treeColor = this.random(cx, cz, i * 10 + 9000) > 0.7 ? 0x1e5631 : 0x228b22;
        const tree = this.createTreeInChunk(baseX + localX, baseZ + localZ, treeColor, height, key);
        if (tree) chunkObjects.push(tree);
      }
      
      // Mushrooms
      const mushroomCount = 3 + Math.floor(this.random(cx, cz, 200) * 5);
      for (let i = 0; i < mushroomCount; i++) {
        const localX = this.random(cx, cz, i * 15 + 1000) * this.chunkSize - this.chunkSize / 2;
        const localZ = this.random(cx, cz, i * 15 + 6000) * this.chunkSize - this.chunkSize / 2;
        const mushroom = this.createMushroomInChunk(baseX + localX, baseZ + localZ, key);
        if (mushroom) chunkObjects.push(mushroom);
      }
    } else if (biome === 'village') {
      // Road through center
      const orientation = this.random(cx, cz, 500) > 0.5 ? 'x' : 'z';
      const road = this.createRoadInChunk(baseX, baseZ, { 
        length: this.chunkSize, 
        width: 12, 
        orientation 
      }, key);
      if (road) chunkObjects.push(road);
      
      // Houses along road
      const houseCount = 2 + Math.floor(this.random(cx, cz, 300) * 3);
      for (let i = 0; i < houseCount; i++) {
        let hx, hz;
        if (orientation === 'x') {
          hx = baseX + (this.random(cx, cz, i * 20) - 0.5) * (this.chunkSize * 0.6);
          hz = baseZ + (this.random(cx, cz, i * 20 + 7000) - 0.5) * this.chunkSize * 0.8;
        } else {
          hx = baseX + (this.random(cx, cz, i * 20 + 7000) - 0.5) * this.chunkSize * 0.8;
          hz = baseZ + (this.random(cx, cz, i * 20) - 0.5) * (this.chunkSize * 0.6);
        }
        const house = this.createHouseInChunk(hx, hz, { 
          width: 10 + this.random(cx, cz, i * 20 + 2000) * 4,
          depth: 9 + this.random(cx, cz, i * 20 + 3000) * 3,
          height: 7 + this.random(cx, cz, i * 20 + 4000) * 4,
          color: Math.floor(this.random(cx, cz, i * 20 + 5000) * 0xffffff),
          roofColor: 0x8a4c38
        }, key);
        if (house) chunkObjects.push(house);
      }
      
      // Street lights
      const lightCount = 2 + Math.floor(this.random(cx, cz, 400) * 3);
      for (let i = 0; i < lightCount; i++) {
        const localX = (this.random(cx, cz, i * 25 + 500) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 25 + 600) - 0.5) * this.chunkSize * 0.9;
        const light = this.createStreetLightInChunk(baseX + localX, baseZ + localZ, key);
        if (light) chunkObjects.push(light);
      }
      
      // Some trees
      const treeCount = 2 + Math.floor(this.random(cx, cz, 450) * 4);
      for (let i = 0; i < treeCount; i++) {
        const localX = (this.random(cx, cz, i * 30 + 700) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 30 + 800) - 0.5) * this.chunkSize * 0.9;
        const tree = this.createTreeInChunk(baseX + localX, baseZ + localZ, 0x228b22, 12, key);
        if (tree) chunkObjects.push(tree);
      }
    } else if (biome === 'plains') {
      // Scattered trees
      const treeCount = 2 + Math.floor(this.random(cx, cz, 550) * 4);
      for (let i = 0; i < treeCount; i++) {
        const localX = (this.random(cx, cz, i * 35 + 900) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 35 + 1100) - 0.5) * this.chunkSize * 0.9;
        const tree = this.createTreeInChunk(baseX + localX, baseZ + localZ, 0x228b22, 14, key);
        if (tree) chunkObjects.push(tree);
      }
      
      // Rocks
      const rockCount = 3 + Math.floor(this.random(cx, cz, 600) * 6);
      for (let i = 0; i < rockCount; i++) {
        const localX = (this.random(cx, cz, i * 40 + 1200) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 40 + 1300) - 0.5) * this.chunkSize * 0.9;
        const rock = this.createRockInChunk(baseX + localX, baseZ + localZ, 0x696969, key);
        if (rock) chunkObjects.push(rock);
      }
    } else if (biome === 'meadow') {
      // Flowers and light vegetation
      const flowerCount = 5 + Math.floor(this.random(cx, cz, 650) * 8);
      for (let i = 0; i < flowerCount; i++) {
        const localX = (this.random(cx, cz, i * 45 + 1400) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 45 + 1500) - 0.5) * this.chunkSize * 0.9;
        const flowerColor = Math.floor(this.random(cx, cz, i * 45 + 1600) * 0xffffff);
        const crystal = this.createCrystalInChunk(baseX + localX, baseZ + localZ, flowerColor, 2, key);
        if (crystal) chunkObjects.push(crystal);
      }
      
      // Few trees
      const treeCount = 1 + Math.floor(this.random(cx, cz, 700) * 3);
      for (let i = 0; i < treeCount; i++) {
        const localX = (this.random(cx, cz, i * 50 + 1700) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 50 + 1800) - 0.5) * this.chunkSize * 0.9;
        const tree = this.createTreeInChunk(baseX + localX, baseZ + localZ, 0x228b22, 11, key);
        if (tree) chunkObjects.push(tree);
      }
    } else { // mystical - rare magical biome
      // Glowing crystals
      const crystalCount = 6 + Math.floor(this.random(cx, cz, 750) * 8);
      for (let i = 0; i < crystalCount; i++) {
        const localX = (this.random(cx, cz, i * 55 + 1900) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 55 + 2000) - 0.5) * this.chunkSize * 0.9;
        const crystalColors = [0xff00ff, 0x00ffff, 0xffff00, 0xff0088, 0x00ff88];
        const crystalColor = crystalColors[Math.floor(this.random(cx, cz, i * 55 + 2100) * crystalColors.length)];
        const height = 4 + this.random(cx, cz, i * 55 + 2200) * 8;
        const crystal = this.createCrystalInChunk(baseX + localX, baseZ + localZ, crystalColor, height, key);
        if (crystal) chunkObjects.push(crystal);
      }
      
      // Floating rocks
      const floatingCount = 2 + Math.floor(this.random(cx, cz, 800) * 4);
      for (let i = 0; i < floatingCount; i++) {
        const localX = (this.random(cx, cz, i * 60 + 2300) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 60 + 2400) - 0.5) * this.chunkSize * 0.9;
        const height = 8 + this.random(cx, cz, i * 60 + 2500) * 10;
        const floating = this.createFloatingRockInChunk(baseX + localX, height, baseZ + localZ, 3, key);
        if (floating) chunkObjects.push(floating);
      }
      
      // Strange purple trees
      const treeCount = 3 + Math.floor(this.random(cx, cz, 850) * 4);
      for (let i = 0; i < treeCount; i++) {
        const localX = (this.random(cx, cz, i * 65 + 2600) - 0.5) * this.chunkSize * 0.9;
        const localZ = (this.random(cx, cz, i * 65 + 2700) - 0.5) * this.chunkSize * 0.9;
        const tree = this.createTreeInChunk(baseX + localX, baseZ + localZ, 0x8b00ff, 15, key);
        if (tree) chunkObjects.push(tree);
      }
      
      // Mystical monoliths
      if (this.random(cx, cz, 900) > 0.7) {
        const localX = (this.random(cx, cz, 2800) - 0.5) * this.chunkSize * 0.6;
        const localZ = (this.random(cx, cz, 2900) - 0.5) * this.chunkSize * 0.6;
        const monolith = this.createMonolith(baseX + localX, baseZ + localZ);
        if (monolith) {
          monolith.userData.chunk = key;
          chunkObjects.push(monolith);
        }
      }
      
      // Portals
      if (this.random(cx, cz, 950) > 0.85) {
        const localX = (this.random(cx, cz, 3000) - 0.5) * this.chunkSize * 0.6;
        const localZ = (this.random(cx, cz, 3100) - 0.5) * this.chunkSize * 0.6;
        const portal = this.createPortal(baseX + localX, baseZ + localZ);
        if (portal) {
          portal.userData.chunk = key;
          chunkObjects.push(portal);
        }
      }
      
      // Floating islands
      if (this.random(cx, cz, 1000) > 0.8) {
        const localX = (this.random(cx, cz, 3200) - 0.5) * this.chunkSize * 0.5;
        const localZ = (this.random(cx, cz, 3300) - 0.5) * this.chunkSize * 0.5;
        const islandY = 25 + this.random(cx, cz, 3400) * 15;
        const island = this.createFloatingIsland(baseX + localX, islandY, baseZ + localZ);
        if (island) {
          island.userData.chunk = key;
          chunkObjects.push(island);
        }
      }
    }
    
    this.chunks.set(key, chunkObjects);
  }
  
  unloadChunk(cx, cz) {
    const key = `${cx},${cz}`;
    const chunkObjects = this.chunks.get(key);
    if (!chunkObjects) return;
    
    chunkObjects.forEach(obj => {
      this.scene.remove(obj);
      this.objects = this.objects.filter(o => o !== obj);
      this.interactableObjects = this.interactableObjects.filter(o => o !== obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
    
    this.chunks.delete(key);
  }
  
  updateChunks() {
    const playerPos = this.camera.position;
    const currentChunkX = Math.floor(playerPos.x / this.chunkSize);
    const currentChunkZ = Math.floor(playerPos.z / this.chunkSize);
    
    // Only update if moved to new chunk
    if (currentChunkX === this.lastChunkUpdate.x && currentChunkZ === this.lastChunkUpdate.z) {
      return;
    }
    
    this.lastChunkUpdate = { x: currentChunkX, z: currentChunkZ };
    
    // Generate chunks in render distance
    for (let x = currentChunkX - this.renderDistance; x <= currentChunkX + this.renderDistance; x++) {
      for (let z = currentChunkZ - this.renderDistance; z <= currentChunkZ + this.renderDistance; z++) {
        this.generateChunk(x, z);
      }
    }
    
    // Unload far chunks
    const toUnload = [];
    this.chunks.forEach((_, key) => {
      const [cx, cz] = key.split(',').map(Number);
      const dx = Math.abs(cx - currentChunkX);
      const dz = Math.abs(cz - currentChunkZ);
      if (dx > this.renderDistance + 1 || dz > this.renderDistance + 1) {
        toUnload.push({ cx, cz });
      }
    });
    
    toUnload.forEach(({ cx, cz }) => this.unloadChunk(cx, cz));
  }

  createTreeInChunk(x, z, color, height, chunkKey) {
    const tree = this.createTree(x, z, color, height);
    if (tree) tree.userData.chunk = chunkKey;
    return tree;
  }
  
  createRockInChunk(x, z, color, chunkKey) {
    const rock = this.createRock(x, z, color);
    if (rock) rock.userData.chunk = chunkKey;
    return rock;
  }
  
  createHouseInChunk(x, z, opts, chunkKey) {
    const house = this.createHouse(x, z, opts);
    if (house) house.userData.chunk = chunkKey;
    return house;
  }
  
  createRoadInChunk(x, z, opts, chunkKey) {
    const road = this.createRoad(x, z, opts);
    if (road) road.userData.chunk = chunkKey;
    return road;
  }
  
  createStreetLightInChunk(x, z, chunkKey) {
    const light = this.createStreetLight(x, z);
    if (light) light.userData.chunk = chunkKey;
    return light;
  }
  
  createMushroomInChunk(x, z, chunkKey) {
    const mushroom = this.createMushroom(x, z);
    if (mushroom) mushroom.userData.chunk = chunkKey;
    return mushroom;
  }
  
  createCrystalInChunk(x, z, color, height, chunkKey) {
    const crystal = this.createCrystal(x, z, color, height);
    if (crystal) crystal.userData.chunk = chunkKey;
    return crystal;
  }
  
  createFloatingRockInChunk(x, y, z, size, chunkKey) {
    const floating = this.createFloatingRock(x, y, z, size);
    if (floating) floating.userData.chunk = chunkKey;
    return floating;
  }
  
  createTree(x, z, color = 0x228b22, height = 15) {
    const group = new THREE.Group();
    group.userData.type = 'tree';
    group.userData.treeHeight = height;
    group.userData.harvestable = true;
    group.userData.woodCount = Math.floor(2 + Math.random() * 3);
    
    // Trunk
    const trunkGeom = new THREE.CylinderGeometry(0.5, 0.8, height * 0.4, 8);
    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x8b4513, roughness: 0.9 });
    const trunk = new THREE.Mesh(trunkGeom, trunkMat);
    trunk.position.y = height * 0.2;
    trunk.castShadow = true;
    group.add(trunk);
    
    // Foliage (cone shape)
    const foliageGeom = new THREE.ConeGeometry(height * 0.3, height * 0.6, 8);
    const foliageMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.8 });
    const foliage = new THREE.Mesh(foliageGeom, foliageMat);
    foliage.position.y = height * 0.6;
    foliage.castShadow = true;
    group.add(foliage);
    
    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createRock(x, z, color = 0x696969) {
    const geom = new THREE.DodecahedronGeometry(Math.random() * 2 + 1, 0);
    const mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.9 });
    const rock = new THREE.Mesh(geom, mat);
    rock.userData.type = 'rock';
    rock.position.set(x, Math.random() * 0.5, z);
    rock.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
    rock.castShadow = true;
    rock.receiveShadow = true;
    this.scene.add(rock);
    this.objects.push(rock);
    this.interactableObjects.push(rock);
    return rock;
  }

  createCrystal(x, z, color = 0x00ffff, height = 5) {
    const geom = new THREE.ConeGeometry(height * 0.2, height, 6);
    const mat = new THREE.MeshStandardMaterial({ 
      color: color, 
      emissive: color,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.8,
      roughness: 0.2,
      metalness: 0.8
    });
    const crystal = new THREE.Mesh(geom, mat);
    crystal.userData.type = 'crystal';
    crystal.userData.height = height;
    crystal.position.set(x, height / 2, z);
    crystal.castShadow = true;
    this.scene.add(crystal);
    this.objects.push(crystal);
    this.interactableObjects.push(crystal);
    
    // Add point light for glow
    const light = new THREE.PointLight(color, 0.5, 10);
    light.position.set(x, height, z);
    this.addLight(light, { castShadow: false });
    
    return crystal;
  }

  createBuilding(x, z, color = 0x4a4a4a, width = 10, height = 30) {
    const geom = new THREE.BoxGeometry(width, height, width);
    const mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.5, metalness: 0.5 });
    const building = new THREE.Mesh(geom, mat);
    building.userData.type = 'building';
    building.userData.building = { width, height };
    building.position.set(x, height / 2, z);
    building.castShadow = true;
    building.receiveShadow = true;
    this.scene.add(building);
    this.objects.push(building);
    this.interactableObjects.push(building);
    
    // Add windows (emissive boxes)
    for (let wy = 3; wy < height - 2; wy += 4) {
      for (let side = 0; side < 4; side++) {
        const windowGeom = new THREE.BoxGeometry(1.5, 2, 0.1);
        const windowMat = new THREE.MeshStandardMaterial({ 
          color: 0xffff88, 
          emissive: 0xffff44,
          emissiveIntensity: Math.random() > 0.3 ? 0.5 : 0
        });
        const windowMesh = new THREE.Mesh(windowGeom, windowMat);
        
        const angle = (side * Math.PI / 2);
        const dist = width / 2 + 0.05;
        windowMesh.position.set(
          x + Math.sin(angle) * dist,
          wy,
          z + Math.cos(angle) * dist
        );
        windowMesh.rotation.y = angle;
        this.scene.add(windowMesh);
        this.objects.push(windowMesh);
      }
    }
    
    return building;
  }

  createHouse(x, z, { width = 12, depth = 10, height = 8, color = 0xb8c4d6, roofColor = 0x8b4f3f } = {}) {
    const house = new THREE.Group();
    house.userData.type = 'house';
    house.userData.house = { width, depth, height };

    // Base box
    const baseGeom = new THREE.BoxGeometry(width, height, depth);
    const baseMat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.1 });
    const base = new THREE.Mesh(baseGeom, baseMat);
    base.position.y = height / 2;
    base.castShadow = true;
    base.receiveShadow = true;
    house.add(base);

    // Roof (simple gable style)
    const roofGeom = new THREE.ConeGeometry(Math.max(width, depth) * 0.55, height * 0.6, 4);
    const roofMat = new THREE.MeshStandardMaterial({ color: roofColor, roughness: 0.7, metalness: 0.05 });
    const roof = new THREE.Mesh(roofGeom, roofMat);
    roof.position.y = height + (height * 0.3);
    roof.rotation.y = Math.PI / 4;
    roof.castShadow = true;
    roof.userData.isHouseRoof = true;
    house.add(roof);

    // Door
    const doorGeom = new THREE.BoxGeometry(width * 0.2, height * 0.35, 0.4);
    const doorMat = new THREE.MeshStandardMaterial({ color: 0x6b4b3a, roughness: 0.6 });
    const door = new THREE.Mesh(doorGeom, doorMat);
    door.position.set(0, height * 0.175, depth / 2 + 0.2);
    door.castShadow = true;
    house.add(door);

    // Simple windows
    const windowGeom = new THREE.BoxGeometry(width * 0.18, height * 0.18, 0.3);
    const windowMat = new THREE.MeshStandardMaterial({ color: 0xaed6ff, emissive: 0x88cfff, emissiveIntensity: 0.2, transparent: true, opacity: 0.85 });
    const windowLeft = new THREE.Mesh(windowGeom, windowMat);
    windowLeft.position.set(-width * 0.25, height * 0.55, depth / 2 + 0.25);
    const windowRight = windowLeft.clone();
    windowRight.position.x = width * 0.25;
    [windowLeft, windowRight].forEach(w => {
      w.castShadow = true;
      house.add(w);
    });

    house.position.set(x, 0, z);
    this.scene.add(house);
    this.objects.push(house);
    this.interactableObjects.push(house);
    return house;
  }

  createRoad(x, z, { length = 200, width = 12, thickness = 0.25, orientation = 'x', color = 0x2f3032 } = {}) {
    const geom = new THREE.BoxGeometry(
      orientation === 'x' ? length : width,
      thickness,
      orientation === 'x' ? width : length
    );
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.9, metalness: 0.15 });
    const road = new THREE.Mesh(geom, mat);
    road.userData.type = 'road';
    road.userData.road = { length, width, thickness, orientation, color };
    road.receiveShadow = true;
    road.position.set(x, thickness * 0.5 + 0.01, z);

    // Center line
    const lineGeom = new THREE.BoxGeometry(
      orientation === 'x' ? length * 0.9 : 0.4,
      thickness * 0.6,
      orientation === 'x' ? 0.4 : width * 0.9
    );
    const lineMat = new THREE.MeshStandardMaterial({ color: 0xf5d142, roughness: 0.8, metalness: 0.05, emissive: 0xf5d142, emissiveIntensity: 0.3 });
    const line = new THREE.Mesh(lineGeom, lineMat);
    line.position.y = thickness * 0.5 + 0.02;
    road.add(line);

    this.scene.add(road);
    this.objects.push(road);
    this.interactableObjects.push(road);
    return road;
  }

  createStreetLight(x, z, { height = 12, color = 0xfff6c4, intensity = 0.9 } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'streetLight';
    group.userData.streetLight = { height, color, intensity };

    const poleGeom = new THREE.CylinderGeometry(0.2, 0.25, height, 8);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.7 });
    const pole = new THREE.Mesh(poleGeom, poleMat);
    pole.position.y = height / 2;
    pole.castShadow = true;
    pole.receiveShadow = true;
    group.add(pole);

    const headGeom = new THREE.BoxGeometry(1.2, 0.6, 1.2);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x777777, roughness: 0.5, metalness: 0.3 });
    const head = new THREE.Mesh(headGeom, headMat);
    head.position.set(0, height, 0);
    group.add(head);

    const light = new THREE.PointLight(color, intensity, 25, 2);
    light.position.set(0, height + 0.6, 0);
    light.castShadow = false; // avoid exceeding max texture units from many shadow maps
    this.addLight(light, { castShadow: false, parent: group });

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createMonolith(x, z, { height = 25, color = 0x2c2c3e, inscriptions = true } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'monolith';
    group.userData.ancient = true;
    group.userData.mysterious = true;

    const geom = new THREE.BoxGeometry(4, height, 6);
    const mat = new THREE.MeshStandardMaterial({ 
      color: color,
      roughness: 0.4,
      metalness: 0.7,
      emissive: 0x6a00ff,
      emissiveIntensity: 0.15
    });
    const pillar = new THREE.Mesh(geom, mat);
    pillar.position.y = height / 2;
    pillar.castShadow = true;
    pillar.receiveShadow = true;
    group.add(pillar);

    if (inscriptions) {
      for (let i = 0; i < 3; i++) {
        const glyphGeom = new THREE.BoxGeometry(0.1, height * 0.15, 1.2);
        const glyphMat = new THREE.MeshStandardMaterial({
          color: 0x00ffff,
          emissive: 0x00ffff,
          emissiveIntensity: 0.8,
          transparent: true,
          opacity: 0.7
        });
        const glyph = new THREE.Mesh(glyphGeom, glyphMat);
        glyph.position.set(2.1, height * 0.25 + i * height * 0.25, 0);
        group.add(glyph);
      }
    }

    const light = new THREE.PointLight(0x6a00ff, 1.2, 30);
    light.position.set(0, height + 2, 0);
    this.addLight(light, { castShadow: false, parent: group });

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createFloatingIsland(x, y, z, { radius = 12, height = 8, rotation = 0 } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'floatingIsland';
    group.userData.floats = true;
    group.userData.originalY = y;

    const geom = new THREE.CylinderGeometry(radius * 0.6, radius, height, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x4a7c5e,
      roughness: 0.9,
      metalness: 0.1
    });
    const island = new THREE.Mesh(geom, mat);
    island.castShadow = true;
    island.receiveShadow = true;
    group.add(island);

    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI * 2;
      const dist = radius * 0.4;
      const treeX = Math.cos(angle) * dist;
      const treeZ = Math.sin(angle) * dist;
      const tree = this.createTree(treeX, treeZ, 0x2d6a3e, 6);
      tree.position.y = height / 2;
      group.add(tree);
      this.scene.remove(tree);
      this.objects = this.objects.filter(o => o !== tree);
      this.interactableObjects = this.interactableObjects.filter(o => o !== tree);
    }

    group.position.set(x, y, z);
    group.rotation.y = rotation;
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createLighthouse(x, z, { height = 40, color = 0xe8e8e8, lightColor = 0xffffff } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'lighthouse';
    group.userData.operational = true;

    const baseGeom = new THREE.CylinderGeometry(4, 5, height * 0.7, 12);
    const baseMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.6 });
    const base = new THREE.Mesh(baseGeom, baseMat);
    base.position.y = height * 0.35;
    base.castShadow = true;
    base.receiveShadow = true;
    group.add(base);

    const topGeom = new THREE.CylinderGeometry(3, 4, height * 0.2, 12);
    const topMat = new THREE.MeshStandardMaterial({ color: 0xcc0000, roughness: 0.5 });
    const top = new THREE.Mesh(topGeom, topMat);
    top.position.y = height * 0.8;
    group.add(top);

    const beaconGeom = new THREE.CylinderGeometry(2.5, 2.5, height * 0.08, 12);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: lightColor,
      emissive: lightColor,
      emissiveIntensity: 1.0,
      transparent: true,
      opacity: 0.9
    });
    const beacon = new THREE.Mesh(beaconGeom, beaconMat);
    beacon.position.y = height * 0.9;
    beacon.userData.rotates = true;
    group.add(beacon);

    const light = new THREE.SpotLight(lightColor, 3, 100, Math.PI / 6, 0.5);
    light.position.set(0, height * 0.9, 0);
    light.target.position.set(50, 0, 0);
    group.add(light.target);
    this.addLight(light, { castShadow: false, parent: group });

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createAncientRuin(x, z, { width = 15, depth = 15, pillars = 6, color = 0x9e8b6d } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'ancientRuin';
    group.userData.historical = true;

    const platformGeom = new THREE.BoxGeometry(width, 1, depth);
    const platformMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.9 });
    const platform = new THREE.Mesh(platformGeom, platformMat);
    platform.position.y = 0.5;
    platform.castShadow = true;
    platform.receiveShadow = true;
    group.add(platform);

    for (let i = 0; i < pillars; i++) {
      const angle = (i / pillars) * Math.PI * 2;
      const radius = width * 0.35;
      const px = Math.cos(angle) * radius;
      const pz = Math.sin(angle) * radius;

      const pillarHeight = 8 + Math.random() * 4;
      const pillarGeom = new THREE.CylinderGeometry(0.6, 0.7, pillarHeight, 8);
      const pillarMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.8 });
      const pillar = new THREE.Mesh(pillarGeom, pillarMat);
      pillar.position.set(px, pillarHeight / 2 + 1, pz);
      pillar.rotation.z = (Math.random() - 0.5) * 0.2;
      pillar.castShadow = true;
      group.add(pillar);
    }

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createPortal(x, z, { height = 12, color1 = 0x8a2be2, color2 = 0x00ffff } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'portal';
    group.userData.magical = true;
    group.userData.destination = null;

    const frameGeom = new THREE.TorusGeometry(height * 0.4, 0.8, 16, 32);
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x1a1a2e,
      emissive: color1,
      emissiveIntensity: 0.5,
      metalness: 0.9,
      roughness: 0.2
    });
    const frame = new THREE.Mesh(frameGeom, frameMat);
    frame.rotation.x = Math.PI / 2;
    frame.position.y = height * 0.5;
    frame.castShadow = true;
    group.add(frame);

    const portalGeom = new THREE.CircleGeometry(height * 0.35, 32);
    const portalMat = new THREE.MeshStandardMaterial({
      color: color2,
      emissive: color2,
      emissiveIntensity: 0.8,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide
    });
    const portal = new THREE.Mesh(portalGeom, portalMat);
    portal.rotation.x = Math.PI / 2;
    portal.position.y = height * 0.5;
    portal.userData.swirls = true;
    group.add(portal);

    const particles = [];
    for (let i = 0; i < 20; i++) {
      const particleGeom = new THREE.SphereGeometry(0.2, 8, 8);
      const particleMat = new THREE.MeshStandardMaterial({
        color: Math.random() > 0.5 ? color1 : color2,
        emissive: Math.random() > 0.5 ? color1 : color2,
        emissiveIntensity: 1.0
      });
      const particle = new THREE.Mesh(particleGeom, particleMat);
      particle.userData.orbits = true;
      particle.userData.angle = (i / 20) * Math.PI * 2;
      particle.userData.radius = height * 0.3;
      particle.userData.speed = 0.5 + Math.random() * 0.5;
      particles.push(particle);
      group.add(particle);
    }
    group.userData.particles = particles;

    const light = new THREE.PointLight(color2, 2, 25);
    light.position.set(0, height * 0.5, 0);
    this.addLight(light, { castShadow: false, parent: group });

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createBridge(x, z, { length = 30, width = 8, orientation = 'x', color = 0x8b7355 } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'bridge';

    const deckGeom = new THREE.BoxGeometry(
      orientation === 'x' ? length : width,
      0.5,
      orientation === 'x' ? width : length
    );
    const deckMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.8 });
    const deck = new THREE.Mesh(deckGeom, deckMat);
    deck.position.y = 0.25;
    deck.castShadow = true;
    deck.receiveShadow = true;
    group.add(deck);

    const railings = 4;
    for (let i = 0; i < railings; i++) {
      const pos = -length / 2 + (i / (railings - 1)) * length;
      const railGeom = new THREE.CylinderGeometry(0.15, 0.15, 3, 8);
      const railMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a });
      
      const rail1 = new THREE.Mesh(railGeom, railMat);
      rail1.position.set(orientation === 'x' ? pos : width / 2 - 0.5, 1.5, orientation === 'x' ? width / 2 - 0.5 : pos);
      
      const rail2 = new THREE.Mesh(railGeom, railMat);
      rail2.position.set(orientation === 'x' ? pos : -width / 2 + 0.5, 1.5, orientation === 'x' ? -width / 2 + 0.5 : pos);
      
      group.add(rail1);
      group.add(rail2);
    }

    group.position.set(x, 5, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createFountain(x, z, { radius = 4, height = 6, color = 0x87ceeb } = {}) {
    const group = new THREE.Group();
    group.userData.type = 'fountain';
    group.userData.flowing = true;

    const basinGeom = new THREE.CylinderGeometry(radius, radius * 1.2, 1, 16);
    const basinMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.6 });
    const basin = new THREE.Mesh(basinGeom, basinMat);
    basin.position.y = 0.5;
    basin.castShadow = true;
    basin.receiveShadow = true;
    group.add(basin);

    const waterGeom = new THREE.CylinderGeometry(radius * 0.9, radius * 1.1, 0.8, 16);
    const waterMat = new THREE.MeshStandardMaterial({
      color: color,
      transparent: true,
      opacity: 0.7,
      roughness: 0.1,
      metalness: 0.3
    });
    const water = new THREE.Mesh(waterGeom, waterMat);
    water.position.y = 0.9;
    group.add(water);

    const spoutGeom = new THREE.CylinderGeometry(0.3, 0.4, height, 8);
    const spoutMat = new THREE.MeshStandardMaterial({ color: 0x8b8b8b });
    const spout = new THREE.Mesh(spoutGeom, spoutMat);
    spout.position.y = height / 2 + 1;
    group.add(spout);

    for (let i = 0; i < 12; i++) {
      const particleGeom = new THREE.SphereGeometry(0.15, 8, 8);
      const particleMat = new THREE.MeshStandardMaterial({
        color: color,
        transparent: true,
        opacity: 0.8
      });
      const particle = new THREE.Mesh(particleGeom, particleMat);
      particle.userData.fountainParticle = true;
      particle.userData.phase = (i / 12) * Math.PI * 2;
      particle.position.y = height + 1;
      group.add(particle);
    }

    group.position.set(x, 0, z);
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    return group;
  }

  createMushroom(x, z, color = 0xff6b6b, height = 2) {
    const group = new THREE.Group();
    group.userData.type = 'mushroom';
    group.userData.mushroomHeight = height;
    
    // Stem
    const stemGeom = new THREE.CylinderGeometry(0.2, 0.3, height * 0.6, 8);
    const stemMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 });
    const stem = new THREE.Mesh(stemGeom, stemMat);
    stem.position.y = height * 0.3;
    group.add(stem);
    
    // Cap
    const capGeom = new THREE.SphereGeometry(height * 0.4, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
    const capMat = new THREE.MeshStandardMaterial({ 
      color: color, 
      emissive: color,
      emissiveIntensity: 0.2,
      roughness: 0.6
    });
    const cap = new THREE.Mesh(capGeom, capMat);
    cap.position.y = height * 0.6;
    group.add(cap);
    
    group.position.set(x, 0, z);
    group.castShadow = true;
    this.scene.add(group);
    this.objects.push(group);
    this.interactableObjects.push(group);
    
    return group;
  }

  createFloatingRock(x, y, z, size = 5) {
    const geom = new THREE.DodecahedronGeometry(size, 1);
    const mat = new THREE.MeshStandardMaterial({ 
      color: 0x8b7355, 
      roughness: 0.9 
    });
    const rock = new THREE.Mesh(geom, mat);
    rock.userData.type = 'floatingRock';
    rock.userData.size = size;
    rock.position.set(x, y, z);
    rock.userData.floatOffset = Math.random() * Math.PI * 2;
    rock.userData.isFloating = true;
    rock.castShadow = true;
    this.scene.add(rock);
    this.objects.push(rock);
    this.interactableObjects.push(rock);
    return rock;
  }

  async callLLM(model, messages, logEl) {
    const setLog = (msg) => {
      if (logEl && typeof logEl.textContent === 'string') {
        logEl.textContent = msg;
      }
    };
    const chosen = (model || 'gpt-oss-20').trim();

    if (chosen.startsWith('web-llm:')) {
      if (!navigator.gpu) {
        throw new Error('WebGPU is not available; pick a local model.');
      }
      if (typeof window === 'undefined' || !window.WebLLMBridge) {
        throw new Error('Browser LLM bridge not loaded yet. Refresh or choose a local model.');
      }
      const modelId = chosen.split(':', 2)[1] || chosen.replace('web-llm:', '');
      setLog(`Loading ${modelId} in browser...`);
      await window.WebLLMBridge.ensureInit(modelId);
      setLog(`Generating with ${modelId} in browser...`);
      return await window.WebLLMBridge.chat(messages);
    }

    setLog('Contacting model...');
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: chosen,
        messages
      })
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      throw new Error(data.error || 'LLM error');
    }
    return data.text || '';
  }

  async generateFromAI() {
    const prompt = document.getElementById('prompt-input').value.trim();
    const model = document.getElementById('model-select').value;
    const statusEl = document.getElementById('status');
    const btn = document.getElementById('generate-btn');
    
    if (!prompt) {
      statusEl.textContent = 'Please enter a description of your world!';
      statusEl.className = 'error';
      return;
    }
    
    btn.disabled = true;
    statusEl.innerHTML = '<span class="loading-spinner"></span>AI is generating your world...';
    statusEl.className = 'loading';
    
    try {
      const systemPrompt = `You are a world generator AI for a 3D game. Based on the user's description, generate a JSON object that defines the world.
      
RESPOND WITH ONLY VALID JSON, no explanations. Use this exact structure:
{
  "skyColor": "#hexcolor",
  "fogColor": "#hexcolor",
  "fogDensity": 0.005,
  "groundColor": "#hexcolor",
  "ambientLight": {"color": "#hexcolor", "intensity": 0.5},
  "sunLight": {"color": "#hexcolor", "intensity": 1, "position": [x, y, z]},
  "trees": [{"x": 0, "z": 0, "color": "#hexcolor", "height": 15}],
  "rocks": [{"x": 0, "z": 0, "color": "#hexcolor"}],
  "crystals": [{"x": 0, "z": 0, "color": "#hexcolor", "height": 5}],
  "buildings": [{"x": 0, "z": 0, "color": "#hexcolor", "width": 10, "height": 30}],
  "houses": [{"x": 0, "z": 0, "color": "#hexcolor", "roofColor": "#hexcolor", "width": 12, "depth": 10, "height": 8}],
  "roads": [{"x": 0, "z": 0, "length": 200, "width": 12, "orientation": "x", "color": "#hexcolor"}],
  "streetLights": [{"x": 0, "z": 0, "color": "#hexcolor", "height": 12, "intensity": 0.9}],
  "mushrooms": [{"x": 0, "z": 0, "color": "#hexcolor", "height": 2}],
  "floatingRocks": [{"x": 0, "y": 20, "z": 0, "size": 5}],
  "pointLights": [{"x": 0, "y": 5, "z": 0, "color": "#hexcolor", "intensity": 1, "distance": 20}]
}

Rules:
- Place roads first; ensure houses sit near roads (offset 8-20 units).
- Keep objects within -100..100 on x and z.
- Generate 25-60 total objects.
- Keep houses away from floating rocks and crystals.
- If theme is urban/suburban, favor houses, roads, street lights, and low fog.
- If theme is natural, reduce roads and lights, keep 1-2 short roads with a few houses.
`;

      const replyText = await this.callLLM(model, [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt }
      ], statusEl);
      let worldData;
      try {
        // Try to extract JSON from the response
        let jsonStr = replyText;
        // Find JSON in the response
        const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          jsonStr = jsonMatch[0];
        }
        worldData = JSON.parse(jsonStr);
      } catch (e) {
        console.error('Failed to parse AI response:', replyText);
        throw new Error('AI response was not valid JSON. Try again!');
      }
      
      this.buildWorldFromData(worldData);
      statusEl.textContent = 'World generated successfully!';
      statusEl.className = 'success';
      
    } catch (error) {
      console.error('Generation error:', error);
      statusEl.textContent = `Error: ${error.message}`;
      statusEl.className = 'error';
    } finally {
      btn.disabled = false;
    }
  }

  buildWorldFromData(data) {
    this.clearWorld();
    
    // Sky and fog
    if (data.skyColor) {
      this.scene.background = new THREE.Color(data.skyColor);
    }
    if (data.fogColor) {
      this.scene.fog = new THREE.FogExp2(new THREE.Color(data.fogColor), data.fogDensity || 0.005);
    }
    
    // Ground
    const groundGeom = new THREE.PlaneGeometry(500, 500, 50, 50);
    const groundMat = new THREE.MeshStandardMaterial({ 
      color: data.groundColor || 0x2d5a27,
      roughness: 0.9,
      metalness: 0.1
    });
    
    const vertices = groundGeom.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      vertices[i + 2] = Math.sin(vertices[i] * 0.03) * Math.cos(vertices[i + 1] * 0.03) * 3;
    }
    groundGeom.computeVertexNormals();
    
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.userData.isGround = true;
    this.scene.add(ground);
    this.ground = ground;
    this.objects.push(ground);
    
    // Ambient light
    const ambientData = data.ambientLight || { color: '#b5d4ff', intensity: 0.85 };
    const ambientIntensity = Math.max(ambientData.intensity || 0.85, 0.7);
    const ambient = new THREE.AmbientLight(new THREE.Color(ambientData.color), ambientIntensity);
    this.addLight(ambient);
    
    // Sun/directional light
    const sunData = data.sunLight || { color: '#ffffff', intensity: 1.2, position: [80, 120, 60] };
    const sunIntensity = Math.max(sunData.intensity || 1.2, 1.0);
    const sun = new THREE.DirectionalLight(new THREE.Color(sunData.color), sunIntensity);
    sun.position.set(...(sunData.position || [80, 120, 60]));
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 10;
    sun.shadow.camera.far = 400;
    sun.shadow.camera.left = -100;
    sun.shadow.camera.right = 100;
    sun.shadow.camera.top = 100;
    sun.shadow.camera.bottom = -100;
    this.addLight(sun, { castShadow: true });
    
    // Trees
    if (data.trees) {
      data.trees.forEach(t => {
        this.createTree(t.x, t.z, new THREE.Color(t.color).getHex(), t.height || 15);
      });
    }
    
    // Rocks
    if (data.rocks) {
      data.rocks.forEach(r => {
        this.createRock(r.x, r.z, new THREE.Color(r.color).getHex());
      });
    }
    
    // Crystals
    if (data.crystals) {
      data.crystals.forEach(c => {
        this.createCrystal(c.x, c.z, new THREE.Color(c.color).getHex(), c.height || 5);
      });
    }
    
    // Buildings
    if (data.buildings) {
      data.buildings.forEach(b => {
        this.createBuilding(b.x, b.z, new THREE.Color(b.color).getHex(), b.width || 10, b.height || 30);
      });
    }

    // Houses
    if (data.houses) {
      data.houses.forEach(h => {
        this.createHouse(h.x, h.z, {
          width: h.width || 12,
          depth: h.depth || 10,
          height: h.height || 8,
          color: new THREE.Color(h.color || 0xb8c4d6).getHex(),
          roofColor: new THREE.Color(h.roofColor || 0x8b4f3f).getHex()
        });
      });
    }

    // Roads
    if (data.roads) {
      data.roads.forEach(r => {
        this.createRoad(r.x || 0, r.z || 0, {
          length: r.length || 200,
          width: r.width || 12,
          orientation: r.orientation || 'x',
          color: new THREE.Color(r.color || 0x2f3032).getHex()
        });
      });
    }

    // Street lights
    if (data.streetLights) {
      data.streetLights.forEach(s => {
        this.createStreetLight(s.x, s.z, {
          height: s.height || 12,
          color: new THREE.Color(s.color || 0xfff6c4).getHex(),
          intensity: s.intensity || 0.9
        });
      });
    }
    
    // Mushrooms
    if (data.mushrooms) {
      data.mushrooms.forEach(m => {
        this.createMushroom(m.x, m.z, new THREE.Color(m.color).getHex(), m.height || 2);
      });
    }
    
    // Floating rocks
    if (data.floatingRocks) {
      data.floatingRocks.forEach(f => {
        this.createFloatingRock(f.x, f.y || 20, f.z, f.size || 5);
      });
    }
    
    // Point lights
    if (data.pointLights) {
      data.pointLights.forEach(p => {
        const light = new THREE.PointLight(
          new THREE.Color(p.color), 
          p.intensity || 1, 
          p.distance || 20
        );
        light.position.set(p.x, p.y || 5, p.z);
        light.castShadow = false; // keep point lights shadowless to stay under texture unit limits
        this.addLight(light, { castShadow: false });
      });
    }
    
    // Reset player position
    this.camera.position.set(0, this.playerHeight, 30);
    this.velocity.set(0, 0, 0);
  }

  // Back-compat loader: accepts AI world JSON, falls back if unsupported
  generateFromData(data) {
    try {
      const d = data || {};
      // Replay primitives path
      if (Array.isArray(d.primitives) || Array.isArray(d.lightPrimitives)) {
        this.buildWorldFromPrimitives(d);
        return;
      }
      // Migration path from legacy saved format (objects/lights) -> primitives
      if (Array.isArray(d.objects) || Array.isArray(d.lights)) {
        const colorStr = (c) => (typeof c === 'number' ? ('#' + c.toString(16).padStart(6, '0')) : (c || '#ffffff'));
        const prims = (d.objects || []).map(o => {
          const gt = (o.type || '').toLowerCase();
          let type = null;
          if (gt.includes('box')) type = 'cube';
          else if (gt.includes('sphere')) type = 'sphere';
          else if (gt.includes('dodecahedron')) type = 'rock';
          else if (gt.includes('cone')) type = 'crystal'; // best-effort mapping; house roofs are grouped, not saved as standalone
          if (!type) return null;
          return {
            type,
            color: colorStr(o.color),
            size: 6,
            x: o.position?.x ?? 0,
            y: o.position?.y ?? 0,
            z: o.position?.z ?? 0
          };
        }).filter(Boolean);
        const lprims = (d.lights || []).map(l => {
          if ((l.type || '').toLowerCase() !== 'pointlight') return null;
          return {
            type: 'light',
            color: colorStr(l.color),
            intensity: l.intensity ?? 1,
            x: l.position?.x ?? 0,
            y: l.position?.y ?? 5,
            z: l.position?.z ?? 0
          };
        }).filter(Boolean);
        this.buildWorldFromPrimitives({ primitives: prims, lightPrimitives: lprims });
        return;
      }
      const looksAIFormat = (
        'skyColor' in d || 'fogColor' in d || 'trees' in d || 'rocks' in d ||
        'crystals' in d || 'buildings' in d || 'houses' in d || 'roads' in d ||
        'streetLights' in d || 'mushrooms' in d || 'floatingRocks' in d || 'pointLights' in d
      );
      if (looksAIFormat) {
        this.buildWorldFromData(d);
        return;
      }
      console.warn('generateFromData: Unsupported worldData format; loading default world');
      this.generateDefaultWorld();
    } catch (err) {
      console.warn('generateFromData failed; loading default world', err);
      this.generateDefaultWorld();
    }
  }

  buildWorldFromPrimitives(d) {
    this.clearWorld();

    // Minimal base: ground + ambient + sun
    const groundGeom = new THREE.PlaneGeometry(500, 500, 10, 10);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x6c7a89, roughness: 0.9, metalness: 0.05 });
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.userData.isGround = true;
    this.scene.add(ground);
    this.ground = ground;
    this.objects.push(ground);

    const ambient = new THREE.AmbientLight(0xb5d4ff, 0.8);
    ambient.name = 'ambient';
    this.addLight(ambient);
    this.ambientLight = ambient;

    const hemi = new THREE.HemisphereLight(0x8fb8ff, 0x2a2416, 0.3);
    hemi.position.set(0, 120, 0);
    hemi.name = 'hemi';
    this.addLight(hemi);
    this.hemiLight = hemi;

    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.name = 'sun';
    sun.position.set(80, 120, 60);
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 10;
    sun.shadow.camera.far = 400;
    sun.shadow.camera.left = -100;
    sun.shadow.camera.right = 100;
    sun.shadow.camera.top = 100;
    sun.shadow.camera.bottom = -100;
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.05;
    sun.shadow.radius = 3;
    this.addLight(sun, { castShadow: true });
    this.sunLight = sun;

    const colorStr = (c) => (typeof c === 'number' ? ('#' + c.toString(16).padStart(6, '0')) : (c || '#ffffff'));

    // Objects
    (d.primitives || []).forEach(p => {
      const type = (p.type || '').toLowerCase();
      const color = colorStr(p.color);
      const size = p.size || 6;
      const x = p.x ?? 0, y = p.y ?? 0, z = p.z ?? 0;
      if (type === 'cube' || type === 'sphere' || type === 'crystal' || type === 'tree' || type === 'rock' || type === 'light' || type === 'mushroom' || type === 'pointlight') {
        this.addObjectAtCoords(type === 'pointlight' ? 'light' : type, color, size, { x, y, z });
      } else if (type === 'streetlight') {
        const height = p.height || Math.max(8, size);
        const intensity = p.intensity ?? 0.9;
        this.createStreetLight(x, z, { height, color, intensity });
      } else if (type === 'building') {
        const width = p.width || Math.max(4, Math.min(size, 50));
        const heightB = p.height || Math.max(6, size);
        this.createBuilding(x, z, new THREE.Color(color).getHex(), width, heightB);
      } else if (type === 'house') {
        const width = p.width || Math.max(6, size);
        const depth = p.depth || Math.max(6, Math.round(size * 0.8));
        const heightH = p.height || Math.max(5, Math.round(size * 0.7));
        const opts = { width, depth, height: heightH, color: new THREE.Color(color).getHex() };
        if (p.roofColor) opts.roofColor = new THREE.Color(colorStr(p.roofColor)).getHex();
        this.createHouse(x, z, opts);
      } else if (type === 'road') {
        const length = p.length || Math.max(20, size * 4);
        const widthR = p.width || Math.max(4, Math.round(size * 0.6));
        const thickness = p.thickness ?? 0.25;
        const orientation = (p.orientation === 'z' ? 'z' : 'x');
        this.createRoad(x, z, { length, width: widthR, thickness, orientation, color: new THREE.Color(color).getHex() });
      } else if (type === 'floatingrock') {
        const sizeFR = p.size || Math.max(3, size);
        this.createFloatingRock(x, y || 10, z, sizeFR);
      }
    });

    // Lights
    (d.lightPrimitives || []).forEach(lp => {
      const color = colorStr(lp.color);
      const intensity = lp.intensity ?? 1.0;
      const light = new THREE.PointLight(new THREE.Color(color), intensity, 20);
      light.userData.type = 'light';
      light.position.set(lp.x ?? 0, lp.y ?? 5, lp.z ?? 0);
      this.addLight(light, { castShadow: false });
      this.interactableObjects.push(light);
    });

    // Reset player position
    this.camera.position.set(0, this.playerHeight, 30);
    this.velocity.set(0, 0, 0);
  }

  getPlacementPoint() {
    const origin = this.camera.position.clone();
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    this.raycaster.set(origin, dir);
    const targets = [...this.interactableObjects];
    if (this.ground) targets.push(this.ground);
    const hit = this.raycaster.intersectObjects(targets, true)[0];
    if (hit) return hit.point.clone();
    return origin.add(dir.multiplyScalar(20));
  }

  addObjectAtPoint(type, colorHex, size = 5) {
    const point = this.getPlacementPoint();
    const color = new THREE.Color(colorHex).getHex();
    switch (type) {
      case 'cube': {
        const geom = new THREE.BoxGeometry(size, size, size);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.type = 'cube';
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.copy(point).setY(point.y + size / 2);
        this.scene.add(mesh);
        this.objects.push(mesh);
        this.interactableObjects.push(mesh);
        break;
      }
      case 'sphere': {
        const geom = new THREE.SphereGeometry(Math.max(1, size / 2), 24, 16);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.type = 'sphere';
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.copy(point).setY(point.y + size / 2);
        this.scene.add(mesh);
        this.objects.push(mesh);
        this.interactableObjects.push(mesh);
        break;
      }
      case 'crystal':
        this.createCrystal(point.x, point.z, color, Math.max(3, size * 0.8));
        break;
      case 'mushroom': {
        const g = this.createMushroom(point.x, point.z, color, Math.max(1, size * 0.5));
        if (g) g.position.y = point.y; // allow placement above ground if ray hits object
        break;
      }
      case 'tree':
        this.createTree(point.x, point.z, color, Math.max(8, size * 1.5));
        break;
      case 'light': {
        const light = new THREE.PointLight(color, 1.2, size * 5);
        light.userData.type = 'light';
        light.position.copy(point).setY(point.y + 3);
        this.addLight(light, { castShadow: false });
        this.interactableObjects.push(light);
        break;
      }
      default:
        this.createRock(point.x, point.z, color);
    }
  }

  addObjectAtCoords(type, colorHex, size = 5, coords = { x: 0, y: 0, z: 0 }) {
    const color = new THREE.Color(colorHex).getHex();
    const { x, y, z } = coords;
    switch (type) {
      case 'cube': {
        const geom = new THREE.BoxGeometry(size, size, size);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.type = 'cube';
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(x, y + size / 2, z);
        this.scene.add(mesh);
        this.objects.push(mesh);
        this.interactableObjects.push(mesh);
        break;
      }
      case 'sphere': {
        const geom = new THREE.SphereGeometry(Math.max(1, size / 2), 24, 16);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.type = 'sphere';
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(x, y + size / 2, z);
        this.scene.add(mesh);
        this.objects.push(mesh);
        this.interactableObjects.push(mesh);
        break;
      }
      case 'crystal':
        this.createCrystal(x, z, color, Math.max(3, size * 0.8));
        break;
      case 'mushroom': {
        const g = this.createMushroom(x, z, color, Math.max(1, size * 0.5));
        if (g) g.position.y = y;
        break;
      }
      case 'tree':
        this.createTree(x, z, color, Math.max(8, size * 1.5));
        break;
      case 'light': {
        const light = new THREE.PointLight(color, 1.2, size * 5);
        light.userData.type = 'light';
        light.position.set(x, y + 3, z);
        this.addLight(light, { castShadow: false });
        this.interactableObjects.push(light);
        break;
      }
      default:
        this.createRock(x, z, color);
    }
  }

  addObjectFromUI() {
    const typeEl = document.getElementById('object-type');
    if (!typeEl) return;
    const colorEl = document.getElementById('object-color');
    const sizeEl = document.getElementById('object-size');
    const type = typeEl.value || 'cube';
    const color = colorEl?.value || '#ff8844';
    const size = parseFloat(sizeEl?.value || '6');
    this.addObjectAtPoint(type, color, size);
  }

  getIntersectedInteractable() {
    this.raycaster.setFromCamera(this._centerVec2, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactableObjects, true).filter(h => h.distance <= this.interactDistance);
    if (!hits.length) return null;
    const target = hits[0].object;
    let obj = target;
    while (obj && !this.interactableObjects.includes(obj)) {
      obj = obj.parent;
    }
    return obj || target;
  }

  placeObjectsFromPlan(rawText) {
    let txt = rawText || '';
    const match = txt.match(/\[[\s\S]*\]/);
    if (match) txt = match[0];
    let items;
    try {
      items = JSON.parse(txt);
    } catch (err) {
      throw new Error('LLM did not return valid JSON');
    }
    if (!Array.isArray(items)) throw new Error('Expected an array of objects');
    let placed = 0;
    items.slice(0, 15).forEach(o => {
      const type = (o.type || 'cube').toLowerCase();
      const color = o.color || '#ffaa55';
      const size = o.size || 6;
      const x = typeof o.x === 'number' ? o.x : (this.camera.position.x + (Math.random() - 0.5) * 20);
      const z = typeof o.z === 'number' ? o.z : (this.camera.position.z + (Math.random() - 0.5) * 20);
      const y = typeof o.y === 'number' ? o.y : 0;
      this.addObjectAtCoords(type, color, size, { x, y, z });
      placed += 1;
    });
    return placed;
  }

  async generateObjectsFromLLM(prompt) {
    const modelSelect = document.getElementById('model-select');
    const model = modelSelect?.value || 'gpt-oss-20';
    const llmLog = document.getElementById('llm-object-log');
    const system = `You are a 3D world builder. Respond ONLY with JSON array of objects. Each object schema: {"type":"cube|sphere|crystal|mushroom|tree|rock|light","color":"#RRGGBB","size":number,"x":number,"z":number,"y":number}. Types: cube(box), sphere(ball), crystal(glowing cone), mushroom(glowing cap), tree(trunk+foliage), rock(irregular), light(point light). Coordinates in range -100..100 for x,z, y default 0. Size: cube/sphere 4-12, crystal/mushroom 2-8, tree 8-20, rock 2-6, light 10-30. Max 15 objects.`;
    const user = `Create objects for this idea: ${prompt}`;
    const replyText = await this.callLLM(model, [
      { role: 'system', content: system },
      { role: 'user', content: user }
    ], llmLog);
    const placed = this.placeObjectsFromPlan(replyText);
    return `Added ${placed} objects via LLM`;
  }

  highlightObject(obj, active) {
    if (!obj) return;
    obj.traverse?.(child => {
      if (child.isMesh && child.material) {
        if (active) {
          child.userData._origEmissive = child.material.emissive ? child.material.emissive.clone() : null;
          child.material.emissive = new THREE.Color(0xffffaa);
          child.material.emissiveIntensity = 0.5;
        } else if (child.userData._origEmissive) {
          child.material.emissive.copy(child.userData._origEmissive);
          child.material.emissiveIntensity = 0.2;
          delete child.userData._origEmissive;
        }
      }
    });
  }

  interactWithHighlighted() {
    if (!this.highlighted) return;
    const obj = this.highlighted;
    const type = obj.userData.type || 'object';
    let removed = false;
    let picked = false;
    
    if (obj.userData.pickupable && type === 'item') {
      picked = this.pickupObject(obj, {
        type: obj.userData.itemType || 'cube',
        color: obj.userData.itemColor || 0xffaa00
      });
    } else if (type === 'tree' && obj.userData.harvestable) {
      const woodCount = obj.userData.woodCount || 3;
      const treePos = obj.position.clone();
      const woodItem = { type: 'wood', color: 0x8b4513 };
      let dropped = false;
      for (let i = 0; i < woodCount; i++) {
        const angle = (Math.PI * 2 * i) / woodCount;
        const radius = 2 + Math.random() * 1.5;
        const wx = treePos.x + Math.cos(angle) * radius;
        const wz = treePos.z + Math.sin(angle) * radius;
        const added = this.addToInventory(woodItem);
        if (!added) {
          this.createItemInWorld(woodItem, wx, wz);
          dropped = true;
        }
      }
      if (dropped) {
        this.setStatus('Inventory full, dropped wood nearby.', 'error');
      } else {
        this.setStatus('Wood gathered!', 'success');
      }
      this.removeWorldObject(obj);
      removed = true;
    } else if (['crystal', 'rock', 'mushroom', 'floatingRock', 'cube', 'sphere'].includes(type)) {
      picked = this.pickupObject(obj, {
        type,
        color: obj.userData.itemColor || obj.material?.color?.getHex?.() || 0xffaa00
      });
    } else if (type === 'light') {
      obj.intensity = obj.intensity > 0 ? 0 : 1.2;
    } else {
      obj.traverse?.(child => {
        if (child.isMesh && child.material?.color) {
          child.material.color.offsetHSL(0.1, 0, 0);
        }
      });
    }
    if (picked) return;
    if (removed) {
      this.worldMemory.score += 10;
      this.displayMemoryStats();
    }
  }

  setCompanionWaypoint(x, z) {
    // Clear previous waypoint marker if exists
    if (this.companion.waypointMarker) {
      this.scene.remove(this.companion.waypointMarker);
    }
    // Set waypoint target
    this.companion.waypoint = { x, z };
    // Create visual marker: magenta glowing sphere
    const markerGeom = new THREE.SphereGeometry(1.5, 16, 12);
    const markerMat = new THREE.MeshStandardMaterial({
      color: 0xff00ff,
      emissive: 0xff00ff,
      emissiveIntensity: 0.8,
      metalness: 0.2,
      roughness: 0.3
    });
    this.companion.waypointMarker = new THREE.Mesh(markerGeom, markerMat);
    this.companion.waypointMarker.position.set(x, 0.5, z);
    this.companion.waypointMarker.castShadow = true;
    this.scene.add(this.companion.waypointMarker);
    // Change behavior to waypoint following
    this.companion.behavior = 'waypoint';
    this.companion.target = { x, z };
    this.setCompanionMessage('📍 Setting waypoint...');
  }

  clearCompanionWaypoint() {
    if (this.companion.waypointMarker) {
      this.scene.remove(this.companion.waypointMarker);
      this.companion.waypointMarker = null;
    }
    this.companion.waypoint = null;
    this.companion.behavior = 'follow';
    this.companion.target = null;
    this.setCompanionMessage('✓ Waypoint cleared.');
  }

  updateAutoLoot(delta) {
    if (!this.companion.mesh || !this.companion.active) return;
    
    // Scan for nearby pickupable items
    const companionPos = this.companion.mesh.position;
    let closestItem = null;
    let closestDist = this.companion.autoLootRange;
    
    this.interactableObjects.forEach(obj => {
      if (obj === this.companion.mesh || !obj.userData?.pickupable) return;
      const dist = companionPos.distanceTo(obj.position);
      if (dist < closestDist) {
        closestItem = obj;
        closestDist = dist;
      }
    });
    
    if (closestItem) {
      // Move companion toward closest item
      const itemPos = closestItem.position;
      const direction = new THREE.Vector3(
        itemPos.x - companionPos.x,
        0,
        itemPos.z - companionPos.z
      ).normalize();
      
      const moveSpeed = Math.min(this.companion.speed * delta, closestDist - 0.5);
      companionPos.add(direction.multiplyScalar(moveSpeed));
      
      // Auto-pickup if close enough
      if (closestDist < 2) {
        const item = this.deriveItemFromObject(closestItem);
        if (this.addToInventory(item)) {
          this.removeWorldObject(closestItem);
          this.companion.message = `📦 Looted ${this.formatItemLabel(item.type)}!`;
        } else {
          this.companion.message = '🧳 Inventory full';
        }
      }
    }
  }

  updateResourceCallouts() {
    if (!this.companion.mesh || !this.companion.active) return;
    
    // Throttle: only check every 3 seconds
    const now = Date.now();
    if (now - this.companion.lastResourceCallout < 3000) return;
    this.companion.lastResourceCallout = now;
    
    // Scan for nearby items within 40 units
    const companionPos = this.companion.mesh.position;
    const resources = {};
    
    this.interactableObjects.forEach(obj => {
      if (obj === this.companion.mesh) return;
      const dist = companionPos.distanceTo(obj.position);
      if (dist > 40 || dist < 2) return; // Ignore too close or too far
      
      const type = obj.userData?.type || 'item';
      const emoji = {
        'wood': '🪵',
        'crystal': '💎',
        'mushroom': '🍄',
        'rock': '🪨',
        'flower': '🌸',
        'food': '🍎'
      }[type] || '📦';
      
      resources[type] = (resources[type] || 0) + 1;
    });
    
    // Announce findings
    if (Object.keys(resources).length > 0) {
      const callout = Object.entries(resources)
        .map(([type, count]) => {
          const emoji = {
            'wood': '🪵',
            'crystal': '💎',
            'mushroom': '🍄',
            'rock': '🪨',
            'flower': '🌸',
            'food': '🍎'
          }[type] || '📦';
          return `${emoji} ${count}`;
        })
        .join(', ');
      this.setCompanionMessage(`Found nearby: ${callout}`);
    }
  }

  createCompanionMesh() {
    const geom = new THREE.SphereGeometry(2, 16, 12);
    const mat = new THREE.MeshStandardMaterial({ color: 0x8ad7ff, emissive: 0x2288ff, emissiveIntensity: 0.4, metalness: 0.1, roughness: 0.4 });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.type = 'companion';
    this.scene.add(mesh);
    return mesh;
  }

  setCompanionMessage(msg) {
    this.companion.message = msg;
    const log = document.getElementById('companion-log');
    if (log) {
      log.textContent = msg;
    }
    this.speakCompanion(msg);
  }

  speakCompanion(text) {
    if (!this.companion.voiceEnabled) return;
    if (typeof window === 'undefined' || !window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') return;
    const cleaned = (text || '').replace(/[\u{1F300}-\u{1FAFF}]/gu, '').trim();
    if (!cleaned) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(cleaned);
      utterance.rate = this.companion.voiceRate;
      utterance.pitch = this.companion.voicePitch;
      utterance.volume = this.companion.voiceVolume;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn('Speech synthesis failed', err);
    }
  }

  async validateOllamaConnection() {
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gpt-oss-20',
          messages: [{ role: 'user', content: 'ping' }]
        })
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        return { ok: false, error: data.error, details: data.details, hint: data.hint };
      }
      return { ok: true };
    } catch (err) {
      return { ok: false, error: 'Connection failed', details: err.message };
    }
  }

  startCompanion() {
    if (!this.companion.mesh) {
      this.companion.mesh = this.createCompanionMesh();
      const startPos = this.camera.position.clone();
      this.companion.mesh.position.copy(startPos).setY(this.playerHeight);
      this.interactableObjects.push(this.companion.mesh);
    }
    this.companion.active = true;
    this.setCompanionMessage('� Checking connection...');
    
    // Test connection and warn if offline
    this.validateOllamaConnection().then(result => {
      if (result.ok) {
        this.setCompanionMessage('👋 Companion online.');
      } else {
        console.warn('Ollama connection error:', result);
        this.setCompanionMessage(`⚠️ ${result.error}: ${result.hint || result.details}`);
      }
    });
    
    this.scheduleCompanionLoop();
  }

  stopCompanion() {
    this.companion.active = false;
    if (this.companion.timer) {
      clearInterval(this.companion.timer);
      this.companion.timer = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    this.setCompanionMessage('⏸ Companion paused.');
  }

  scheduleCompanionLoop() {
    if (this.companion.timer) clearInterval(this.companion.timer);
    this.companion.timer = setInterval(() => this.pollCompanionLLM(), 5000);
    this.pollCompanionLLM();
  }

  sampleWorldState() {
    const playerPos = this.camera.position;
    const items = this.interactableObjects
      .filter(o => o !== this.companion.mesh)
      .slice(0, 8)
      .map(o => ({
        type: o.userData?.type || o.type || 'object',
        x: Number(o.position.x.toFixed(1)),
        z: Number(o.position.z.toFixed(1))
      }));
    return {
      player: {
        x: Number(playerPos.x.toFixed(1)),
        y: Number(playerPos.y.toFixed(1)),
        z: Number(playerPos.z.toFixed(1))
      },
      items,
      last_message: this.companion.message
    };
  }

  async pollCompanionLLM() {
    if (!this.companion.active) return;
    const state = this.sampleWorldState();
    const model = this.companion.model || 'gpt-oss-20';
    
    // Add personality context
    const personalities = {
      friendly: 'cheerful and supportive',
      curious: 'inquisitive and exploratory',
      protective: 'cautious and watchful'
    };
    const personalityDesc = personalities[this.companion.personality] || 'friendly';
    
    // Include recent memory
    const recentMemory = this.companion.memory.slice(-3).map(m => m.event).join('. ');
    const memoryContext = recentMemory ? ` Recent events: ${recentMemory}.` : '';
    
    const system = `You are a ${personalityDesc} in-world AI companion. Respond ONLY with JSON. Schema: {"action":"move|follow|explore|guard","target":[x,z],"message":"short text","speed":30,"emotion":"happy|curious|alert|calm"}. Rules: stay within 80 units of player, be helpful and ${personalityDesc}, keep messages short (<=60 chars), adapt behavior to situation.${memoryContext}`;
    const user = `Player and world state: ${JSON.stringify(state)}. Biome: ${this.getBiome(Math.floor(state.player.x / this.chunkSize), Math.floor(state.player.z / this.chunkSize))}`;
    
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user }
          ]
        })
      });
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || 'LLM error');
      let txt = data.text || '';
      const match = txt.match(/\{[\s\S]*\}/);
      if (match) txt = match[0];
      const parsed = JSON.parse(txt);
      
      if (parsed.message) {
        const emotion = parsed.emotion || '💬';
        const emotionIcons = { happy: '😊', curious: '🤔', alert: '⚠️', calm: '😌' };
        const icon = emotionIcons[emotion] || '💬';
        this.setCompanionMessage(`${icon} ${parsed.message}`);
        
        // Store in memory
        this.companion.memory.push({
          time: Date.now(),
          event: parsed.message,
          action: parsed.action
        });
        if (this.companion.memory.length > 10) this.companion.memory.shift();
      }
      
      // Check for active waypoint first
      if (this.companion.waypoint) {
        const wpDist = Math.sqrt(
          (this.companion.mesh.position.x - this.companion.waypoint.x) ** 2 +
          (this.companion.mesh.position.z - this.companion.waypoint.z) ** 2
        );
        if (wpDist < 5) {
          // Waypoint reached
          this.clearCompanionWaypoint();
          this.setCompanionMessage('✅ Waypoint reached!');
        } else {
          // Continue toward waypoint
          this.companion.target = this.companion.waypoint;
          this.companion.behavior = 'waypoint';
          return; // Skip LLM decision while on waypoint
        }
      }
      
      // Smarter pathfinding
      if (parsed.action === 'follow' || parsed.action === 'guard') {
        const offset = parsed.action === 'guard' ? 15 : 8;
        const angle = Math.random() * Math.PI * 2;
        this.companion.target = { 
          x: state.player.x + Math.cos(angle) * offset, 
          z: state.player.z + Math.sin(angle) * offset 
        };
        this.companion.behavior = parsed.action;
      } else if (parsed.action === 'explore' && parsed.target && Array.isArray(parsed.target) && parsed.target.length >= 2) {
        this.companion.target = { x: parsed.target[0], z: parsed.target[1] };
        this.companion.behavior = 'explore';
      } else if (parsed.target && Array.isArray(parsed.target) && parsed.target.length >= 2) {
        this.companion.target = { x: parsed.target[0], z: parsed.target[1] };
      } else {
        // Default follow behavior
        this.companion.target = { x: state.player.x, z: state.player.z };
        this.companion.behavior = 'follow';
      }
      
      if (parsed.speed) this.companion.speed = Math.max(10, Math.min(100, parsed.speed));
    } catch (err) {
      const errorMsg = err?.message || 'Unknown error';
      console.error('Companion LLM error:', errorMsg);
      
      // Provide specific error feedback
      if (errorMsg.includes('Ollama error')) {
        this.setCompanionMessage('⚠️ Ollama offline. Start with: ollama serve');
      } else if (errorMsg.includes('Failed to fetch')) {
        this.setCompanionMessage('⚠️ Server not responding');
      } else if (errorMsg.includes('JSON')) {
        this.setCompanionMessage('⚠️ Invalid response format');
      } else {
        this.setCompanionMessage(`⚠️ Connection lost...`);
      }
      
      // Keep companion moving in default follow pattern
      const state = this.sampleWorldState();
      this.companion.target = { x: state.player.x, z: state.player.z };
      this.companion.behavior = 'follow';
    }
  }

  createResidentMesh() {
    const body = new THREE.Group();
    const torso = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 3.2, 1.6),
      new THREE.MeshStandardMaterial({ color: 0x9ae6b4, roughness: 0.55, metalness: 0.05 })
    );
    torso.position.y = 2.3;
    torso.castShadow = true;
    torso.receiveShadow = true;

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(1.1, 18, 12),
      new THREE.MeshStandardMaterial({ color: 0xbde0fe, emissive: 0x3b82f6, emissiveIntensity: 0.18 })
    );
    head.position.y = 4.5;
    head.castShadow = true;
    head.receiveShadow = true;

    const eyeGeom = new THREE.BoxGeometry(0.15, 0.35, 0.15);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, emissive: 0x0f172a, emissiveIntensity: 0.6 });
    const leftEye = new THREE.Mesh(eyeGeom, eyeMat);
    const rightEye = new THREE.Mesh(eyeGeom, eyeMat);
    leftEye.position.set(-0.35, 4.65, 0.95);
    rightEye.position.set(0.35, 4.65, 0.95);

    body.add(torso, head, leftEye, rightEye);
    body.userData.type = 'resident';
    return body;
  }

  spawnResident() {
    if (this.resident.mesh) return;
    this.resident.mesh = this.createResidentMesh();
    const startPos = this.camera.position.clone();
    this.resident.mesh.position.copy(startPos).add(new THREE.Vector3(4, 0, 4));
    this.resident.mesh.position.y = this.playerHeight;
    this.resident.target = { x: startPos.x + 8, z: startPos.z + 2 };
    this.scene.add(this.resident.mesh);
    this.interactableObjects.push(this.resident.mesh);
    this.updateResidentUI('Settling into the world...', 'curious');
  }

  tickResidentBrain(delta) {
    if (!this.resident.mesh) return;
    // simple needs model
    this.resident.hunger = Math.min(100, this.resident.hunger + delta * 2);
    this.resident.energy = Math.max(0, this.resident.energy - delta * 1.5);

    const playerPos = this.controls?.getObject?.().position || this.camera.position;
    const resPos = this.resident.mesh.position;
    const distToPlayer = resPos.distanceTo(playerPos);

    // decide mood
    if (this.resident.hunger > 80) this.resident.mood = 'hungry';
    else if (this.resident.energy < 25) this.resident.mood = 'tired';
    else if (distToPlayer > 40) this.resident.mood = 'lonely';
    else this.resident.mood = 'curious';

    this.updateResidentUI(
      `Hunger ${Math.round(this.resident.hunger)} · Energy ${Math.round(this.resident.energy)} · ${this.resident.state}`,
      this.resident.mood
    );

    this.resident.timer -= delta;
    if (this.resident.timer <= 0) {
      this.resident.timer = 4 + Math.random() * 3;

      // choose action
      if (this.resident.hunger > 70) {
        this.resident.state = 'foraging';
        const angle = Math.random() * Math.PI * 2;
        this.resident.target = {
          x: playerPos.x + Math.cos(angle) * 12,
          z: playerPos.z + Math.sin(angle) * 12
        };
      } else if (this.resident.energy < 20) {
        this.resident.state = 'resting';
        this.resident.target = { x: resPos.x, z: resPos.z };
        this.resident.energy = Math.min(100, this.resident.energy + 15);
      } else if (distToPlayer > 30) {
        this.resident.state = 'seeking company';
        this.resident.target = { x: playerPos.x, z: playerPos.z };
      } else {
        this.resident.state = 'wandering';
        const angle = Math.random() * Math.PI * 2;
        const radius = 10 + Math.random() * 18;
        this.resident.target = {
          x: playerPos.x + Math.cos(angle) * radius,
          z: playerPos.z + Math.sin(angle) * radius
        };
      }
    }
  }

  updateResident(delta) {
    if (!this.resident.mesh || !this.resident.target) return;
    const pos = this.resident.mesh.position;
    const target = this.resident.target;
    const dir = this._tmpVec3a.set(target.x - pos.x, 0, target.z - pos.z);
    const dist = dir.length();

    if (dist < 0.8) {
      // recover a bit when reaching spot
      this.resident.hunger = Math.max(0, this.resident.hunger - 8 * delta);
      this.resident.energy = Math.min(100, this.resident.energy + 6 * delta);
      return;
    }

    dir.normalize();
    const playerPos = this.controls?.getObject?.().position || this.camera.position;
    const minDistToPlayer = 4;
    const speed = this.resident.state === 'resting' ? 4 : 18;
    const maxStep = Math.max(0.01, speed * delta);
    const step = Math.min(maxStep, Math.max(0, dist - 0.8));

    // Move toward target
    const nextX = pos.x + dir.x * step;
    const nextZ = pos.z + dir.z * step;

    // Keep a small separation from the player
    const dxp = nextX - playerPos.x;
    const dzp = nextZ - playerPos.z;
    const d2p = (dxp * dxp) + (dzp * dzp);
    if (d2p < (minDistToPlayer * minDistToPlayer)) {
      // push away from player slightly
      const push = this._tmpVec3b.set(dxp, 0, dzp);
      const plen = push.length() || 1;
      push.multiplyScalar((minDistToPlayer - Math.sqrt(d2p)) / plen);
      pos.x = nextX + push.x;
      pos.z = nextZ + push.z;
    } else {
      pos.x = nextX;
      pos.z = nextZ;
    }
    pos.y = this.playerHeight;
  }

  updateCompanion(delta) {
    if (!this.companion.active || !this.companion.mesh) {
      return;
    }
    
    // Update auto-loot and resource callouts
    this.updateAutoLoot(delta);
    this.updateResourceCallouts();
    
    // If no target, return early
    if (!this.companion.target) return;
    
    const pos = this.companion.mesh.position;
    const target = this.companion.target;
    const dir = this._tmpVec3a.set(target.x - pos.x, 0, target.z - pos.z);
    const dist = dir.length();
    if (dist < 0.8) return;
    dir.normalize();

    // Clamp speed and avoid clipping through player
    const playerPos = this.controls?.getObject?.().position || this.camera.position;
    const minDistToPlayer = 3;
    const speed = Math.max(5, Math.min(120, this.companion.speed || 40));
    const maxStep = speed * delta;
    const step = Math.min(maxStep, Math.max(0, dist - 0.8));

    const nextX = pos.x + dir.x * step;
    const nextZ = pos.z + dir.z * step;
    const dxp = nextX - playerPos.x;
    const dzp = nextZ - playerPos.z;
    const d2p = (dxp * dxp) + (dzp * dzp);
    if (d2p < (minDistToPlayer * minDistToPlayer)) {
      const push = this._tmpVec3b.set(dxp, 0, dzp);
      const plen = push.length() || 1;
      push.multiplyScalar((minDistToPlayer - Math.sqrt(d2p)) / plen);
      pos.x = nextX + push.x;
      pos.z = nextZ + push.z;
    } else {
      pos.x = nextX;
      pos.z = nextZ;
    }
    pos.y = this.playerHeight; // keep companion at player height
  }

  saveWorldMemory() {
    // Save current world state to localStorage
    try {
      const pos = this.camera.position;
      const vel = this.velocity;
      
      this.worldMemory.playerPosition = { x: pos.x, y: pos.y, z: pos.z };
      this.worldMemory.playerVelocity = { x: vel.x, y: vel.y, z: vel.z };
      this.worldMemory.playtime += (Date.now() - this.worldMemory.sessionStart) / 1000;
      this.worldMemory.sessionStart = Date.now();
      
      // Save to localStorage
      localStorage.setItem('worldMemory', JSON.stringify(this.worldMemory));
      
      console.log('💾 World memory saved:', {
        position: this.worldMemory.playerPosition,
        score: this.worldMemory.score,
        playtime: Math.floor(this.worldMemory.playtime),
        sessions: this.worldMemory.sessions
      });
    } catch (err) {
      console.error('Failed to save world memory:', err);
    }
  }

  loadWorldMemory() {
    // Load world state from localStorage
    try {
      const saved = localStorage.getItem('worldMemory');
      if (saved) {
        const data = JSON.parse(saved);
        this.worldMemory = {
          ...this.worldMemory,
          ...data,
          sessionStart: Date.now(),
          sessions: (data.sessions || 0) + 1
        };
        
        // Restore player position
        if (this.worldMemory.playerPosition && this.camera) {
          const pos = this.worldMemory.playerPosition;
          this.camera.position.set(pos.x, pos.y, pos.z);
        }
        
        // Restore velocity
        if (this.worldMemory.playerVelocity) {
          const vel = this.worldMemory.playerVelocity;
          this.velocity.set(vel.x, vel.y, vel.z);
        }
        
        // Restore inventory
        if (Array.isArray(this.worldMemory.inventory)) {
          this.inventory = this.worldMemory.inventory;
        }
        
        console.log('📂 World memory loaded:', {
          position: this.worldMemory.playerPosition,
          score: this.worldMemory.score,
          playtime: Math.floor(this.worldMemory.playtime),
          sessions: this.worldMemory.sessions
        });
        
        this.displayMemoryStats();
      } else {
        console.log('🆕 Starting new world memory');
      }
    } catch (err) {
      console.error('Failed to load world memory:', err);
    }

    this.updatePlayerUI(true);
    this.updateInventoryUI();
  }

  clearWorldMemory() {
    // Clear all saved world memory
    localStorage.removeItem('worldMemory');
    this.worldMemory = {
      playerPosition: { x: 0, y: 6, z: 30 },
      playerVelocity: { x: 0, y: 0, z: 0 },
      worldData: null,
      score: 0,
      playtime: 0,
      sessionStart: Date.now(),
      sessions: 0,
      inventory: []
    };
    console.log('🗑️ World memory cleared');
  }

  saveCurrentWorld() {
    // Save the current world configuration
    const worldData = {
      timestamp: Date.now(),
      objects: [],
      lights: [],
      // New replay-friendly format
      primitives: [],
      lightPrimitives: []
    };

    // Helper: hex number -> css string
    const hexToString = (c) => {
      if (typeof c === 'number') return '#' + c.toString(16).padStart(6, '0');
      return c || '#ffffff';
    };
    // Helper: approximate size via bounding box
    const getApproxSize = (obj) => {
      const box = new THREE.Box3().setFromObject(obj);
      const size = new THREE.Vector3();
      box.getSize(size);
      return Math.max(size.x, size.y, size.z) || 1;
    };
    // Helper: find a representative color on object or its children
    const getObjectColorHex = (obj) => {
      if (obj.material?.color?.getHex) return obj.material.color.getHex();
      let found = null;
      if (obj.traverse) {
        obj.traverse(child => {
          if (!found && child.isMesh && child.material?.color?.getHex) {
            found = child.material.color.getHex();
          }
        });
      }
      return found ?? 0xffffff;
    };
    // Helper: build primitive record
    const toPrimitive = (obj) => {
      const ptype = obj.userData?.type || null;
      let type = ptype;
      if (!type) {
        const gtype = obj.geometry?.type || '';
        if (gtype.includes('Box')) type = 'cube';
        else if (gtype.includes('Sphere')) type = 'sphere';
        else if (gtype.includes('Dodecahedron')) type = 'rock';
      }
      if (!type) return null;
      const allowed = ['cube','sphere','tree','rock','light','crystal','mushroom','streetLight','building','house','road','floatingRock'];
      if (!allowed.includes(type)) return null;
      const colorHex = getObjectColorHex(obj);
      const size = getApproxSize(obj);
      const base = {
        type,
        color: hexToString(colorHex),
        size: Math.round(size),
        x: obj.position.x,
        y: obj.position.y,
        z: obj.position.z
      };
      // Specialized enrichment
      if (type === 'crystal' && typeof obj.userData?.height === 'number') {
        base.size = Math.round(obj.userData.height);
      }
      if (type === 'tree' && typeof obj.userData?.treeHeight === 'number') {
        base.size = Math.round(obj.userData.treeHeight);
      }
      if (type === 'mushroom' && typeof obj.userData?.mushroomHeight === 'number') {
        base.size = Math.round(obj.userData.mushroomHeight);
      }
      if (type === 'streetLight') {
        const meta = obj.userData?.streetLight || {};
        base.height = meta.height || size;
        base.intensity = meta.intensity ?? 0.9;
        base.color = hexToString(meta.color ?? colorHex);
      }
      if (type === 'building') {
        const meta = obj.userData?.building || {};
        base.width = meta.width || size;
        base.height = meta.height || size;
      }
      if (type === 'house') {
        const meta = obj.userData?.house || {};
        base.width = meta.width || size;
        base.depth = meta.depth || size;
        base.height = meta.height || Math.round(size * 0.7);
        // try to pick roof color
        let roofColor = null;
        obj.traverse?.(child => {
          if (!roofColor && child.userData?.isHouseRoof && child.material?.color?.getHex) {
            roofColor = hexToString(child.material.color.getHex());
          }
        });
        if (roofColor) base.roofColor = roofColor;
      }
      if (type === 'road') {
        const meta = obj.userData?.road || {};
        if (meta.length && meta.width) {
          base.length = meta.length;
          base.width = meta.width;
          base.thickness = meta.thickness ?? 0.25;
          base.orientation = meta.orientation || 'x';
          base.color = hexToString(meta.color ?? colorHex);
        } else {
          // approximate from bounding box
          const box = new THREE.Box3().setFromObject(obj);
          const s = new THREE.Vector3();
          box.getSize(s);
          base.length = Math.max(s.x, s.z);
          base.width = Math.min(s.x, s.z);
          base.thickness = s.y;
          base.orientation = s.x >= s.z ? 'x' : 'z';
        }
      }
      if (type === 'floatingRock') {
        base.size = obj.userData?.size || Math.round(size);
      }
      return base;
    };

    // Extract raw objects (back-compat)
    this.objects.forEach(obj => {
      const data = {
        type: obj.geometry?.type || 'unknown',
        position: { x: obj.position.x, y: obj.position.y, z: obj.position.z },
        rotation: { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z },
        color: obj.material?.color?.getHex() || 0xffffff
      };
      worldData.objects.push(data);
    });

    // Extract replay primitives from interactables to avoid ground/roads clutter
    const seen = new Set();
    this.interactableObjects.forEach(obj => {
      if (seen.has(obj)) return;
      seen.add(obj);
      const prim = toPrimitive(obj);
      if (prim) worldData.primitives.push(prim);
    });

    // Extract light data (back-compat) and light primitives
    this.lights.forEach(light => {
      const data = {
        type: light.type,
        position: { x: light.position.x, y: light.position.y, z: light.position.z },
        color: light.color?.getHex() || 0xffffff,
        intensity: light.intensity
      };
      worldData.lights.push(data);
      const isStreetLightBulb = !!(light.parent && light.parent.userData && light.parent.userData.type === 'streetLight');
      if (!isStreetLightBulb && (light.userData?.type === 'light' || light.type === 'PointLight')) {
        worldData.lightPrimitives.push({
          type: 'light',
          color: hexToString(data.color),
          intensity: data.intensity,
          x: data.position.x,
          y: data.position.y,
          z: data.position.z
        });
      }
    });

    this.worldMemory.worldData = worldData;
    this.saveWorldMemory();
    console.log('🌍 World configuration saved:', worldData.objects.length, 'objects');
  }

  formatPlaytime(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }

  updatePlayerUI(force = false) {
    if (!this.ui || !this.ui.pos) return;
    const now = performance.now();
    if (!force && now - this.lastUIUpdate < 150) return;
    this.lastUIUpdate = now;

    const playerPos = this.controls?.getObject?.().position || this.camera.position;
    this.ui.pos.textContent = `${playerPos.x.toFixed(1)}, ${playerPos.y.toFixed(1)}, ${playerPos.z.toFixed(1)}`;

    const horizontalSpeed = Math.sqrt((this.velocity.x ** 2) + (this.velocity.z ** 2));
    this.ui.speed.textContent = horizontalSpeed.toFixed(1);

    const sessions = this.worldMemory?.sessions || 0;
    this.ui.session.textContent = `${sessions}`;

    const elapsed = (Date.now() - (this.worldMemory?.sessionStart || Date.now())) / 1000;
    const totalPlaytime = (this.worldMemory?.playtime || 0) + Math.max(elapsed, 0);
    this.ui.playtime.textContent = this.formatPlaytime(totalPlaytime);

    // Update resident panel if alive
    if (this.resident?.mesh) {
      this.updateResidentUI(
        `Hunger ${Math.round(this.resident.hunger)} · Energy ${Math.round(this.resident.energy)} · ${this.resident.state}`,
        this.resident.mood
      );
    }
  }

  displayMemoryStats() {
    // Display world memory statistics in UI
    const stats = document.getElementById('memory-stats');
    if (stats) {
      const elapsed = (Date.now() - (this.worldMemory.sessionStart || Date.now())) / 1000;
      const totalPlaytime = (this.worldMemory.playtime || 0) + Math.max(elapsed, 0);
      const playtimeLabel = this.formatPlaytime(totalPlaytime);
      const pos = this.controls?.getObject?.().position || this.camera?.position || { x: 0, y: 0, z: 0 };
      stats.innerHTML = `
        <div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; z-index: 1000;">
          <strong>📊 World Memory</strong><br>
          Sessions: ${this.worldMemory.sessions}<br>
          Playtime: ${playtimeLabel}<br>
          Score: ${this.worldMemory.score}<br>
          Position: (${Math.floor(pos.x)}, ${Math.floor(pos.y)}, ${Math.floor(pos.z)})
        </div>
      `;
    }
  }

  startAutoSave(intervalSeconds = 30) {
    // Start automatic world saving
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }
    this.autoSaveInterval = setInterval(() => {
      this.saveWorldMemory();
    }, intervalSeconds * 1000);
    console.log(`🔄 Auto-save enabled (every ${intervalSeconds}s)`);
  }

  stopAutoSave() {
    // Stop automatic world saving
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
      this.autoSaveInterval = null;
      console.log('⏸️ Auto-save disabled');
    }
  }

  updateTimeOfDay(delta) {
    // Update day/night cycle
    this.timeOfDay += this.timeSpeed * delta * 1000;
    if (this.timeOfDay > 1) this.timeOfDay = 0;
    
    // Update sky and lighting based on time
    const sunPhase = Math.sin(this.timeOfDay * Math.PI);
    const sunIntensity = sunPhase * 0.8 + 0.4;
    const ambientIntensity = sunPhase * 0.5 + 0.4;
    
    // Adjust sky color based on time
    let skyColor;
    if (this.timeOfDay < 0.25) { // Night
      skyColor = new THREE.Color(0x0a0a1e);
    } else if (this.timeOfDay < 0.35) { // Dawn
      skyColor = new THREE.Color(0xff6b35);
    } else if (this.timeOfDay < 0.65) { // Day
      skyColor = new THREE.Color(0x87ceeb);
    } else if (this.timeOfDay < 0.75) { // Dusk
      skyColor = new THREE.Color(0xff4500);
    } else { // Night
      skyColor = new THREE.Color(0x0a0a1e);
    }

    if (this.timeOfDay < 0.25) {
      this._sunColor.setHex(0x1a2a4a);
      this._ambientColor.setHex(0x111827);
    } else if (this.timeOfDay < 0.35) {
      this._sunColor.setHex(0xffb67a);
      this._ambientColor.setHex(0xffd6a5);
    } else if (this.timeOfDay < 0.65) {
      this._sunColor.setHex(0xffffff);
      this._ambientColor.copy(skyColor);
    } else if (this.timeOfDay < 0.75) {
      this._sunColor.setHex(0xff8a4c);
      this._ambientColor.setHex(0xffc7a3);
    } else {
      this._sunColor.setHex(0x1a2a4a);
      this._ambientColor.setHex(0x111827);
    }

    const sunLight = this.sunLight || this.lights.find(light => light.type === 'DirectionalLight');
    const ambientLight = this.ambientLight || this.lights.find(light => light.type === 'AmbientLight');
    const hemiLight = this.hemiLight || this.lights.find(light => light.type === 'HemisphereLight');

    if (sunLight) {
      sunLight.intensity = sunIntensity;
      if (sunLight.color) sunLight.color.copy(this._sunColor);
    }
    if (ambientLight) {
      ambientLight.intensity = ambientIntensity;
      if (ambientLight.color) ambientLight.color.copy(this._ambientColor);
    }
    if (hemiLight) {
      hemiLight.intensity = ambientIntensity * 0.6;
      if (hemiLight.color) hemiLight.color.copy(this._ambientColor);
      if (hemiLight.groundColor) {
        hemiLight.groundColor.copy(this._ambientColor).lerp(this._blackColor, 0.3);
      }
    }

    if (this.renderer && 'toneMappingExposure' in this.renderer) {
      this.renderer.toneMappingExposure = 0.85 + sunPhase * 0.35;
    }

    if (this.scene.background && this.scene.background.isColor) {
      this.scene.background.lerp(skyColor, 0.04);
    } else {
      this.scene.background = skyColor.clone();
    }
    if (this.scene.fog && this.scene.fog.color) {
      this.scene.fog.color.lerp(skyColor, 0.04);
      if ('density' in this.scene.fog) {
        this.scene.fog.density = 0.003 + (1 - sunPhase) * 0.002;
      }
    }
  }

  updateWeather(delta) {
    // Update weather particles
    this.particles.forEach((particle, index) => {
      if (this.weatherType === 'rain') {
        particle.position.y -= 40 * delta;
        particle.position.x += Math.sin(Date.now() * 0.001) * 0.5 * delta;
      } else if (this.weatherType === 'snow') {
        particle.position.y -= 8 * delta;
        particle.position.x += Math.sin(Date.now() * 0.002 + index) * 2 * delta;
        particle.position.z += Math.cos(Date.now() * 0.002 + index) * 2 * delta;
      }
      
      // Respawn particles that fall too low
      if (particle.position.y < this.camera.position.y - 20) {
        particle.position.y = this.camera.position.y + 30;
        particle.position.x = this.camera.position.x + (Math.random() - 0.5) * 100;
        particle.position.z = this.camera.position.z + (Math.random() - 0.5) * 100;
      }
    });
  }

  setWeather(type) {
    // Change weather type and create particles
    this.weatherType = type;
    
    // Remove old particles
    this.particles.forEach(p => this.scene.remove(p));
    this.particles = [];
    
    if (type === 'clear') return;
    
    // Create new weather particles
    const particleCount = type === 'rain' ? 500 : 300;
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
      color: type === 'rain' ? 0x4444ff : 0xffffff,
      size: type === 'rain' ? 0.3 : 0.8,
      transparent: true,
      opacity: type === 'rain' ? 0.6 : 0.8
    });
    
    const positions = [];
    for (let i = 0; i < particleCount; i++) {
      positions.push(
        this.camera.position.x + (Math.random() - 0.5) * 100,
        this.camera.position.y + (Math.random() - 0.5) * 60,
        this.camera.position.z + (Math.random() - 0.5) * 100
      );
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const particleSystem = new THREE.Points(geometry, material);
    this.scene.add(particleSystem);
    this.particles.push(particleSystem);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    
    // Safety check: ensure renderer exists
    if (!this.renderer || !this.scene || !this.camera) {
      return;
    }
    
    const time = performance.now();
    const delta = (time - this.prevTime) / 1000;
    
    // Animate floating objects
    this.objects.forEach(obj => {
      if (obj.userData && obj.userData.isFloating) {
        obj.position.y += Math.sin(time * 0.001 + obj.userData.floatOffset) * 0.01;
        obj.rotation.y += 0.002;
      }
      
      // Animate floating islands
      if (obj.userData && obj.userData.floats) {
        const originalY = obj.userData.originalY || 0;
        obj.position.y = originalY + Math.sin(time * 0.0005) * 2;
        obj.rotation.y += 0.0003;
      }
      
      // Animate portal particles
      if (obj.userData && obj.userData.particles) {
        obj.userData.particles.forEach(particle => {
          if (particle.userData.orbits) {
            particle.userData.angle += particle.userData.speed * delta;
            const radius = particle.userData.radius;
            particle.position.x = Math.cos(particle.userData.angle) * radius;
            particle.position.z = Math.sin(particle.userData.angle) * radius;
          }
        });
        
        // Rotate portal surface
        const portal = obj.children.find(c => c.userData.swirls);
        if (portal) portal.rotation.z += 0.5 * delta;
      }
      
      // Animate lighthouse beacon
      if (obj.userData && obj.userData.operational) {
        const beacon = obj.children.find(c => c.userData.rotates);
        if (beacon) beacon.rotation.y += 0.8 * delta;
      }
      
      // Animate fountain particles
      obj.children.forEach(child => {
        if (child.userData.fountainParticle) {
          const t = time * 0.002 + child.userData.phase;
          child.position.y = 6 + Math.sin(t) * 3;
          child.position.x = Math.cos(t * 0.5) * 0.8;
          child.position.z = Math.sin(t * 0.5) * 0.8;
        }
      });
    });

    // Hover detection at crosshair
    const target = this.getIntersectedInteractable();
    if (target !== this.highlighted) {
      if (this.highlighted) this.highlightObject(this.highlighted, false);
      this.highlighted = target;
      if (this.highlighted) this.highlightObject(this.highlighted, true);
    }
    
    if (this.controls.isLocked) {
      // Apply friction
      this.velocity.x -= this.velocity.x * 10.0 * delta;
      this.velocity.z -= this.velocity.z * 10.0 * delta;
      
      // Gravity
      this.velocity.y -= 800 * delta;
      
      // Movement direction
      this.direction.z = Number(this.moveForward) - Number(this.moveBackward);
      this.direction.x = Number(this.moveRight) - Number(this.moveLeft);
      this.direction.normalize();
      
      const speed = this.isRunning ? 800 : 400;
      
      if (this.moveForward || this.moveBackward) {
        this.velocity.z -= this.direction.z * speed * delta;
      }
      if (this.moveLeft || this.moveRight) {
        this.velocity.x -= this.direction.x * speed * delta;
      }
      
      this.controls.moveRight(-this.velocity.x * delta);
      this.controls.moveForward(-this.velocity.z * delta);
      
      this.controls.getObject().position.y += this.velocity.y * delta;
      
      // Ground collision
      if (this.controls.getObject().position.y < this.playerHeight) {
        this.velocity.y = 0;
        this.controls.getObject().position.y = this.playerHeight;
        this.canJump = true;
      }
    }

    // Companion movement
    this.updateCompanion(delta);

    // Resident AI movement and needs
    this.tickResidentBrain(delta);
    this.updateResident(delta);

    // Update chunks for infinite world
    this.updateChunks();
    
    // Update day/night cycle
    this.updateTimeOfDay(delta);
    
    // Update weather effects
    this.updateWeather(delta);

    // Update enhanced world (dynamic lighting, particles, biomes)
    if (this.enhanced) {
      this.enhanced.update(delta);
    }

    // Sync sky/environment with current lighting
    this.syncAtmosphere();

    // Player HUD
    this.updatePlayerUI();
    
    this.prevTime = time;
    
    // Final safety check before rendering
    if (this.renderer && this.scene && this.camera) {
      try {
        this.renderer.render(this.scene, this.camera);
      } catch (err) {
        console.error('Render error:', err);
        this.setStatus('Render failed: ' + err.message, 'error');
      }
    }
  }
}

// Initialize the game
window.addEventListener('load', () => {
  console.log('🎮 Initializing World Generator...');
  
  try {
    window.game = new WorldGenerator();
    console.log('✓ World Generator initialized');
    
    // Start auto-save
    window.game.startAutoSave(30);
    
    // Create memory stats display
    const statsDiv = document.createElement('div');
    statsDiv.id = 'memory-stats';
    document.body.appendChild(statsDiv);
    window.game.displayMemoryStats();
    
    // Update stats periodically
    setInterval(() => {
      window.game.displayMemoryStats();
    }, 5000);
    
    console.log('✓ Game fully initialized and running');
  } catch (err) {
    console.error('❌ Failed to initialize game:', err);
    const statusEl = document.getElementById('status');
    if (statusEl) {
      statusEl.textContent = 'Failed to initialize: ' + err.message;
      statusEl.style.color = '#ef4444';
    }
  }
});

// ============================================================================
// LLM CHAT INTEGRATION
// ============================================================================

class LLMChat {
  constructor() {
    this.chatPanel = document.getElementById('llm-chat-panel');
    this.chatMessages = document.getElementById('chat-messages');
    this.chatInput = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('send-chat-btn');
    this.toggleBtn = document.getElementById('toggle-chat-btn');
    this.closeBtn = document.getElementById('close-chat-btn');
    this.modelName = document.getElementById('chat-model-name');
    
    this.messages = [];
    this.maxHistory = 10;
    this.isProcessing = false;
    
    this.init();
  }
  
  init() {
    // Toggle chat panel
    this.toggleBtn.addEventListener('click', () => {
      const isHidden = this.chatPanel.style.display === 'none';
      this.chatPanel.style.display = isHidden ? 'block' : 'none';
      this.toggleBtn.style.display = isHidden ? 'none' : 'block';
      if (isHidden) this.chatInput.focus();
    });
    
    // Close chat
    this.closeBtn.addEventListener('click', () => {
      this.chatPanel.style.display = 'none';
      this.toggleBtn.style.display = 'block';
    });
    
    // Send message
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    
    // Keyboard shortcut: C to open chat
    document.addEventListener('keydown', (e) => {
      if (e.key === 'c' && !e.ctrlKey && !e.altKey && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (this.chatPanel.style.display === 'none') {
          this.chatPanel.style.display = 'block';
          this.toggleBtn.style.display = 'none';
          this.chatInput.focus();
        }
      }
    });
    
    console.log('✓ LLM Chat initialized (press C to open chat)');
  }
  
  async sendMessage() {
    const userInput = this.chatInput.value.trim();
    if (!userInput || this.isProcessing) return;
    
    // Add user message
    this.addMessage('user', userInput);
    this.chatInput.value = '';
    this.isProcessing = true;
    this.sendBtn.textContent = '...';
    this.sendBtn.disabled = true;
    
    try {
      // Call the server API with conversation history
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: 'You are a friendly AI companion in a 3D game world. Keep responses brief, helpful, and engaging. You help the player explore, learn, and have fun.' },
            ...this.messages.map(m => ({ role: m.role, content: m.content }))
          ],
          model: 'gpt-oss-20' // Use the local model
        })
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      const reply = data.text || 'Sorry, I couldn\'t generate a response.';
      
      // Add assistant message
      this.addMessage('assistant', reply);
      
      // Speak the response if companion has voice enabled
      if (window.game && window.game.companion.voiceEnabled) {
        this.speak(reply);
      }
      
    } catch (err) {
      console.error('Chat error:', err);
      this.addMessage('system', `Error: ${err.message}. Make sure the server is running with 'npm run dev'.`);
    } finally {
      this.isProcessing = false;
      this.sendBtn.textContent = 'Send';
      this.sendBtn.disabled = false;
    }
  }
  
  addMessage(role, content) {
    // Add to history
    this.messages.push({ role, content });
    
    // Trim history
    if (this.messages.length > this.maxHistory * 2) {
      this.messages = this.messages.slice(-this.maxHistory * 2);
    }
    
    // Clear placeholder if first message
    if (this.messages.length === 1) {
      this.chatMessages.innerHTML = '';
    }
    
    // Create message element
    const msgDiv = document.createElement('div');
    msgDiv.style.marginBottom = '12px';
    msgDiv.style.padding = '10px 12px';
    msgDiv.style.borderRadius = '8px';
    
    if (role === 'user') {
      msgDiv.style.background = 'rgba(34, 211, 238, 0.15)';
      msgDiv.style.borderLeft = '3px solid var(--accent)';
      msgDiv.style.color = 'var(--text)';
      msgDiv.innerHTML = `<strong style="color: var(--accent);">You:</strong><br>${this.escapeHtml(content)}`;
    } else if (role === 'assistant') {
      msgDiv.style.background = 'rgba(139, 92, 246, 0.15)';
      msgDiv.style.borderLeft = '3px solid #8b5cf6';
      msgDiv.style.color = 'var(--text)';
      msgDiv.innerHTML = `<strong style="color: #a78bfa;">AI Companion:</strong><br>${this.escapeHtml(content)}`;
    } else {
      msgDiv.style.background = 'rgba(245, 158, 11, 0.15)';
      msgDiv.style.borderLeft = '3px solid var(--accent-2)';
      msgDiv.style.color = 'var(--muted)';
      msgDiv.style.fontSize = '12px';
      msgDiv.textContent = content;
    }
    
    this.chatMessages.appendChild(msgDiv);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }
  
  speak(text) {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = window.game.companion.voiceRate || 1.02;
      utterance.pitch = window.game.companion.voicePitch || 1.05;
      utterance.volume = window.game.companion.voiceVolume || 0.85;
      
      window.speechSynthesis.speak(utterance);
    }
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
  }
}

// Initialize chat when page loads
window.addEventListener('load', () => {
  setTimeout(() => {
    window.llmChat = new LLMChat();
    console.log('✓ LLM Chat ready');
  }, 1000);
});
      statusEl.textContent = 'Initialization failed: ' + err.message;
      statusEl.className = 'error';
    }
  }
});

// Save on page unload
window.addEventListener('beforeunload', () => {
  if (window.game) {
    window.game.saveWorldMemory();
  }
});
