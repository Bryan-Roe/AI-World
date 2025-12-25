/**
 * Enhanced World System - Improvements Module
 * Adds: Enhanced particles, dynamic lighting, biome transitions, post-processing
 */

export class ParticleSystem {
  constructor(scene) {
    this.scene = scene;
    this.systems = [];
  }

  addRain(intensity = 0.8) {
    const count = Math.floor(500 * intensity);
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 200;
      positions[i * 3 + 1] = Math.random() * 150 + 50;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0x888888,
      size: 0.2,
      transparent: true,
      opacity: 0.6,
    });

    const particles = new THREE.Points(geometry, material);
    particles.userData = { type: 'rain', speed: 80 };
    this.scene.add(particles);
    this.systems.push(particles);
    return particles;
  }

  addSnow(intensity = 0.6) {
    const count = Math.floor(300 * intensity);
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 200;
      positions[i * 3 + 1] = Math.random() * 150 + 50;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 200;

      velocities[i * 3] = (Math.random() - 0.5) * 0.3;
      velocities[i * 3 + 1] = -Math.random() * 0.5 - 0.3;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

    const material = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.4,
      transparent: true,
      opacity: 0.8,
    });

    const particles = new THREE.Points(geometry, material);
    particles.userData = { type: 'snow', velocities };
    this.scene.add(particles);
    this.systems.push(particles);
    return particles;
  }

  addMist(intensity = 0.5) {
    const count = Math.floor(200 * intensity);
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 200;
      positions[i * 3 + 1] = Math.random() * 50 + 10;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xcccccc,
      size: 1.0,
      transparent: true,
      opacity: 0.2,
    });

    const particles = new THREE.Points(geometry, material);
    particles.userData = { type: 'mist', speed: 5 };
    this.scene.add(particles);
    this.systems.push(particles);
    return particles;
  }

  update(delta) {
    for (const system of this.systems) {
      const positions = system.geometry.attributes.position.array;

      if (system.userData.type === 'rain') {
        for (let i = 0; i < positions.length; i += 3) {
          positions[i + 1] -= system.userData.speed * delta;
          if (positions[i + 1] < 0) {
            positions[i + 1] = 200;
          }
        }
      } else if (system.userData.type === 'snow') {
        const velocities = system.userData.velocities;
        for (let i = 0; i < positions.length; i += 3) {
          positions[i] += velocities[i] * delta;
          positions[i + 1] += velocities[i + 1] * delta;
          positions[i + 2] += velocities[i + 2] * delta;

          if (positions[i + 1] < 0) {
            positions[i + 1] = 200;
            positions[i] = (Math.random() - 0.5) * 200;
          }
        }
      } else if (system.userData.type === 'mist') {
        for (let i = 0; i < positions.length; i += 3) {
          positions[i] += Math.sin(Date.now() * 0.0005 + i) * 0.1;
        }
      }

      system.geometry.attributes.position.needsUpdate = true;
    }
  }

  clear() {
    for (const system of this.systems) {
      this.scene.remove(system);
    }
    this.systems = [];
  }
}

export class LightingController {
  constructor(scene) {
    this.scene = scene;
    this.time = 0.5; // 0=night, 0.5=day, 1=night
    this.speed = 0.0001;
    this.sun = null;
    this.ambient = null;
  }

  initialize() {
    // Remove old lights if they exist
    const oldSun = this.scene.getObjectByName('sun');
    const oldAmbient = this.scene.getObjectByName('ambient');
    if (oldSun) this.scene.remove(oldSun);
    if (oldAmbient) this.scene.remove(oldAmbient);

    // Create sun
    this.sun = new THREE.DirectionalLight(0xffffff, 1.2);
    this.sun.name = 'sun';
    this.sun.position.set(100, 120, 80);
    this.scene.add(this.sun);

    // Create ambient
    this.ambient = new THREE.AmbientLight(0xb5d4ff, 0.85);
    this.ambient.name = 'ambient';
    this.scene.add(this.ambient);
  }

  update(delta) {
    this.time += this.speed * delta;
    if (this.time > 1) this.time -= 1;

    // Sun rotation
    const angle = this.time * Math.PI * 2;
    const sunHeight = Math.sin(angle);
    const sunDistance = Math.cos(angle);

    this.sun.position.set(sunDistance * 100, Math.max(10, sunHeight * 120 + 60), 80);

    // Adjust light intensity based on time
    const intensity = Math.max(0.2, sunHeight + 0.5);
    this.sun.intensity = intensity * 1.5;
    this.ambient.intensity = 0.4 + intensity * 0.5;

    // Sky color
    const skyColor = new THREE.Color();
    if (sunHeight < 0) {
      // Night
      skyColor.setHSL(0.6, 0.2, 0.15);
    } else if (sunHeight < 0.3) {
      // Dawn/Dusk
      skyColor.setHSL(0.08, 0.8, 0.4);
    } else {
      // Day
      skyColor.setHSL(0.55, 0.7, 0.5 + sunHeight * 0.3);
    }

    this.scene.background = skyColor;
    this.scene.fog.color = skyColor;
  }
}

export class BiomeTransition {
  constructor(scene) {
    this.scene = scene;
    this.currentBiome = 'forest';
    this.biomes = {
      forest: { color: 0x228B22, fogColor: 0x4a7c4e },
      village: { color: 0x8B7355, fogColor: 0x8B7355 },
      plains: { color: 0xADFF2F, fogColor: 0xb8ff47 },
      desert: { color: 0xEDC9AF, fogColor: 0xf5d6a8 },
      mystic: { color: 0x663399, fogColor: 0x8b4dba },
    };
  }

  transitionTo(biomeName, duration = 2000) {
    if (!this.biomes[biomeName]) return;

    const targetBiome = this.biomes[biomeName];
    const startColor = this.scene.background.getHex();
    const targetColor = new THREE.Color(targetBiome.color);
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      const color = new THREE.Color(startColor);
      color.lerp(targetColor, progress);
      this.scene.background = color;

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        this.currentBiome = biomeName;
      }
    };

    animate();
  }
}

export class PostProcessing {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.composer = null;
    this.bloomPass = null;
  }

  initialize() {
    const EffectComposer = window.THREE.EffectComposer || this.createEffectComposer();
    if (!EffectComposer) return;

    this.composer = new EffectComposer(this.renderer);
    const renderPass = new THREE.RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    // Add bloom for magical glow
    const bloomPass = new THREE.UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.5, // strength
      0.4, // radius
      0.85 // threshold
    );
    this.bloomPass = bloomPass;
    this.composer.addPass(bloomPass);
  }

  render() {
    if (this.composer) {
      this.composer.render();
    }
  }

  createEffectComposer() {
    // Fallback if EffectComposer not available
    console.warn('EffectComposer not available, skipping post-processing');
    return null;
  }
}

export class EnhancedWorld {
  constructor(worldGenerator) {
    this.world = worldGenerator;
    this.particles = new ParticleSystem(worldGenerator.scene);
    this.lighting = new LightingController(worldGenerator.scene);
    this.biome = new BiomeTransition(worldGenerator.scene);
    this.postProcessing = null;

    this.initialize();
  }

  initialize() {
    this.lighting.initialize();

    // Try to setup post-processing if available
    if (window.THREE && window.THREE.EffectComposer) {
      this.postProcessing = new PostProcessing(
        this.world.renderer,
        this.world.scene,
        this.world.camera
      );
      this.postProcessing.initialize();
    }
  }

  setWeather(type) {
    this.particles.clear();
    if (type === 'rain') {
      this.particles.addRain(0.8);
    } else if (type === 'snow') {
      this.particles.addSnow(0.6);
    } else if (type === 'mist') {
      this.particles.addMist(0.5);
    }
  }

  setBiome(biomeName) {
    this.biome.transitionTo(biomeName, 1500);
  }

  setTimeSpeed(speed) {
    this.lighting.speed = speed;
  }

  update(delta) {
    this.lighting.update(delta);
    this.particles.update(delta);

    if (this.postProcessing) {
      this.postProcessing.render();
    }
  }
}
