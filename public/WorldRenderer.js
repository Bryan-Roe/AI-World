/**
 * Advanced World Renderer
 * Renders AI-generated worlds with materials, lighting, and effects
 */

export class WorldRenderer {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.objects = [];
    this.lights = [];
    this.materials = new Map();
  }

  /**
   * Load and render a world from JSON
   */
  async renderWorld(worldData) {
    console.log('🌍 Rendering world:', worldData.name);
    
    // Clear previous world
    this.clearWorld();
    
    // Set atmosphere
    this.setAtmosphere(worldData);
    
    // Create terrain
    if (worldData.terrain_objects) {
      this.createTerrain(worldData.terrain_objects);
    }
    
    // Create structures
    if (worldData.structures) {
      this.createStructures(worldData.structures);
    }
    
    // Create vegetation
    if (worldData.vegetation) {
      this.createVegetation(worldData.vegetation);
    }
    
    // Create lights
    if (worldData.lights) {
      this.createLights(worldData.lights);
    }
    
    // Add environment effects
    if (worldData.weather) {
      this.addWeatherEffects(worldData.weather);
    }
    
    console.log(`✅ World rendered with ${this.objects.length} objects and ${this.lights.length} lights`);
    
    return {
      objectCount: this.objects.length,
      lightCount: this.lights.length,
      name: worldData.name
    };
  }

  /**
   * Set world atmosphere (fog, sky, lighting)
   */
  setAtmosphere(worldData) {
    // Fog
    const fogColor = worldData.fog_color ? parseInt(worldData.fog_color.replace('0x', ''), 16) : 0x87ceeb;
    const fogDensity = worldData.fog_density || 0.005;
    this.scene.fog = new THREE.FogExp2(fogColor, fogDensity);
    
    // Sky color
    const skyColor = parseInt(worldData.sky_color.replace('0x', ''), 16);
    this.scene.background = new THREE.Color(skyColor);
    
    // Atmosphere settings
    if (worldData.atmosphere) {
      const ambient = new THREE.AmbientLight(0xffffff, worldData.atmosphere.ambientLight || 0.5);
      this.scene.add(ambient);
      this.lights.push(ambient);
    }
  }

  /**
   * Create terrain objects with proper materials
   */
  createTerrain(terrainObjects) {
    for (const terrainObj of terrainObjects) {
      let geometry;
      
      switch(terrainObj.type) {
        case 'plane':
          geometry = new THREE.PlaneGeometry(terrainObj.scale[0], terrainObj.scale[2]);
          break;
        case 'cylinder':
          geometry = new THREE.CylinderGeometry(
            terrainObj.scale[0] / 2, 
            terrainObj.scale[0] / 2, 
            terrainObj.scale[1],
            32
          );
          break;
        case 'sphere':
          geometry = new THREE.SphereGeometry(terrainObj.scale[0] / 2, 32, 32);
          break;
        default:
          continue;
      }
      
      // Create material with physical properties
      const material = new THREE.MeshStandardMaterial({
        color: parseInt(terrainObj.color.replace('0x', ''), 16),
        roughness: terrainObj.roughness || 0.7,
        metalness: terrainObj.metalness || 0.1,
        side: THREE.DoubleSide
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...terrainObj.position);
      mesh.castShadow = true;
      mesh.receiveShadow = terrainObj.receives_shadow !== false;
      
      this.scene.add(mesh);
      this.objects.push(mesh);
    }
  }

  /**
   * Create structures with proper materials and shadows
   */
  createStructures(structures) {
    for (const struct of structures) {
      let geometry;
      
      const geometryMap = {
        'cube': () => new THREE.BoxGeometry(struct.scale[0], struct.scale[1], struct.scale[2]),
        'sphere': () => new THREE.SphereGeometry(struct.scale[0] / 2, 32, 32),
        'cylinder': () => new THREE.CylinderGeometry(struct.scale[0] / 2, struct.scale[0] / 2, struct.scale[1], 32),
        'cone': () => new THREE.ConeGeometry(struct.scale[0] / 2, struct.scale[1], 32),
        'torus': () => new THREE.TorusGeometry(struct.scale[0] / 2, struct.scale[0] / 4, 16, 100),
        'pyramid': () => new THREE.ConeGeometry(struct.scale[0] / 2, struct.scale[1], 4),
        'octahedron': () => new THREE.OctahedronGeometry(struct.scale[0] / 2),
        'tetrahedron': () => new THREE.TetrahedronGeometry(struct.scale[0] / 2),
        'house': () => this.createHouseGeometry(struct.scale),
        'tower': () => this.createTowerGeometry(struct.scale),
        'bridge': () => this.createBridgeGeometry(struct.scale),
        'arch': () => this.createArchGeometry(struct.scale),
        'pillar': () => this.createCylinderGeometry(struct.scale),
        'statue': () => new THREE.ConeGeometry(struct.scale[0] / 3, struct.scale[1], 32),
        'temple': () => this.createTempleGeometry(struct.scale),
        'fortress': () => this.createFortressGeometry(struct.scale),
      };
      
      const geometryFn = geometryMap[struct.type] || geometryMap['cube'];
      geometry = geometryFn();
      
      // Create material
      const material = new THREE.MeshStandardMaterial({
        color: parseInt(struct.color.replace('0x', ''), 16),
        roughness: struct.roughness || 0.5,
        metalness: struct.metalness || 0.2,
        emissive: parseInt(struct.emissive?.replace('0x', '') || '0', 16),
        emissiveIntensity: struct.emissive && struct.emissive !== '0x000000' ? 0.3 : 0
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...struct.position);
      mesh.rotation.z = (struct.rotation || 0) * Math.PI / 180;
      mesh.castShadow = struct.casts_shadow !== false;
      mesh.receiveShadow = true;
      
      this.scene.add(mesh);
      this.objects.push(mesh);
    }
  }

  /**
   * Create vegetation with animation support
   */
  createVegetation(vegetation) {
    for (const veg of vegetation) {
      let geometry;
      
      const vegMap = {
        'tree': () => this.createTreeGeometry(veg.scale),
        'bush': () => new THREE.SphereGeometry(veg.scale[0], 16, 16),
        'grass': () => this.createGrassGeometry(veg.scale),
        'flower': () => this.createFlowerGeometry(veg.scale),
        'cactus': () => this.createCactusGeometry(veg.scale),
        'coral': () => this.createCoralGeometry(veg.scale),
        'crystal': () => this.createCrystalGeometry(veg.scale),
        'mushroom': () => this.createMushroomGeometry(veg.scale)
      };
      
      const geometryFn = vegMap[veg.type] || vegMap['bush'];
      geometry = geometryFn();
      
      const material = new THREE.MeshStandardMaterial({
        color: parseInt(veg.color.replace('0x', ''), 16),
        roughness: 0.8,
        metalness: 0,
        side: THREE.DoubleSide
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...veg.position);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      
      // Store animation type for later
      mesh.userData.animation = veg.animation || 'none';
      mesh.userData.basePosition = [...veg.position];
      
      this.scene.add(mesh);
      this.objects.push(mesh);
    }
  }

  /**
   * Create lights based on world data
   */
  createLights(lightsData) {
    for (const lightData of lightsData) {
      const color = parseInt(lightData.color.replace('0x', ''), 16);
      const intensity = lightData.intensity || 1.0;
      
      let light;
      
      switch(lightData.type) {
        case 'ambient':
          light = new THREE.AmbientLight(color, intensity);
          break;
        
        case 'directional':
          light = new THREE.DirectionalLight(color, intensity);
          if (lightData.direction) {
            light.position.set(lightData.direction[0] * 100, lightData.direction[1] * 100, lightData.direction[2] * 100);
          }
          light.castShadow = true;
          light.shadow.mapSize.set(2048, 2048);
          light.shadow.camera.far = 200;
          break;
        
        case 'point':
          light = new THREE.PointLight(color, intensity, lightData.distance || 100);
          if (lightData.position) {
            light.position.set(...lightData.position);
          }
          if (lightData.decay) {
            light.decay = lightData.decay;
          }
          light.castShadow = true;
          break;
        
        case 'spot':
          light = new THREE.SpotLight(color, intensity, lightData.distance || 200, lightData.angle || 1, lightData.penumbra || 0.5, 1);
          if (lightData.position) {
            light.position.set(...lightData.position);
          }
          if (lightData.direction) {
            const target = new THREE.Vector3(
              light.position.x + lightData.direction[0],
              light.position.y + lightData.direction[1],
              light.position.z + lightData.direction[2]
            );
            light.target.position.copy(target);
          }
          light.castShadow = true;
          break;
        
        case 'hemisphere':
          light = new THREE.HemisphereLight(color, 0x444444, intensity);
          break;
      }
      
      if (light) {
        this.scene.add(light);
        if (light.target) {
          this.scene.add(light.target);
        }
        this.lights.push(light);
      }
    }
  }

  /**
   * Add weather effects to the scene
   */
  addWeatherEffects(weatherType) {
    // This can be extended to add particle systems, sounds, etc.
    console.log(`🌤️  Weather: ${weatherType}`);
  }

  /**
   * Animate the world (vegetation swaying, etc.)
   */
  animate(time) {
    for (const obj of this.objects) {
      if (obj.userData.animation === 'sway') {
        const offset = Math.sin(time * 0.001) * 0.1;
        obj.position.y = obj.userData.basePosition[1] + offset;
      } else if (obj.userData.animation === 'pulse') {
        const scale = 1 + Math.sin(time * 0.002) * 0.05;
        obj.scale.set(scale, scale, scale);
      }
    }
  }

  /**
   * Helper: Create tree geometry
   */
  createTreeGeometry(scale) {
    const group = new THREE.Group();
    
    // Trunk
    const trunk = new THREE.CylinderGeometry(scale[0] * 0.2, scale[0] * 0.25, scale[1] * 0.4, 8);
    const trunkMesh = new THREE.Mesh(trunk, new THREE.MeshStandardMaterial({ color: 0x8b4513 }));
    trunkMesh.position.y = scale[1] * 0.2;
    group.add(trunkMesh);
    
    // Canopy
    const canopy = new THREE.SphereGeometry(scale[0] * 0.4, 16, 16);
    const canopyMesh = new THREE.Mesh(canopy, new THREE.MeshStandardMaterial({ color: 0x228b22 }));
    canopyMesh.position.y = scale[1] * 0.6;
    group.add(canopyMesh);
    
    return group;
  }

  /**
   * Helper: Create flower geometry
   */
  createFlowerGeometry(scale) {
    const group = new THREE.Group();
    const petalCount = 5;
    
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2;
      const petal = new THREE.SphereGeometry(scale[0] * 0.2, 8, 8);
      const petalMesh = new THREE.Mesh(petal);
      petalMesh.position.x = Math.cos(angle) * scale[0] * 0.3;
      petalMesh.position.z = Math.sin(angle) * scale[0] * 0.3;
      group.add(petalMesh);
    }
    
    // Center
    const center = new THREE.SphereGeometry(scale[0] * 0.15, 8, 8);
    const centerMesh = new THREE.Mesh(center);
    group.add(centerMesh);
    
    return group;
  }

  /**
   * Helper: Create grass geometry
   */
  createGrassGeometry(scale) {
    const geometry = new THREE.ConeGeometry(scale[0] * 0.1, scale[1], 4);
    return geometry;
  }

  /**
   * Helper: Create house geometry
   */
  createHouseGeometry(scale) {
    const group = new THREE.Group();
    
    // Base
    const base = new THREE.BoxGeometry(scale[0], scale[1] * 0.7, scale[2]);
    const baseMesh = new THREE.Mesh(base);
    group.add(baseMesh);
    
    // Roof (cone)
    const roof = new THREE.ConeGeometry(Math.max(scale[0], scale[2]) * 0.6, scale[1] * 0.5, 4);
    const roofMesh = new THREE.Mesh(roof);
    roofMesh.position.y = scale[1] * 0.6;
    group.add(roofMesh);
    
    return group;
  }

  /**
   * Helper: Create tower geometry
   */
  createTowerGeometry(scale) {
    return new THREE.CylinderGeometry(scale[0] * 0.3, scale[0] * 0.3, scale[1], 16);
  }

  /**
   * Helper: Create bridge geometry
   */
  createBridgeGeometry(scale) {
    return new THREE.BoxGeometry(scale[0], scale[1] * 0.2, scale[2]);
  }

  /**
   * Helper: Create arch geometry
   */
  createArchGeometry(scale) {
    const group = new THREE.Group();
    
    // Left pillar
    const pillar = new THREE.CylinderGeometry(scale[0] * 0.2, scale[0] * 0.2, scale[1], 8);
    const left = new THREE.Mesh(pillar);
    left.position.x = -scale[0] * 0.4;
    group.add(left);
    
    // Right pillar
    const right = new THREE.Mesh(pillar);
    right.position.x = scale[0] * 0.4;
    group.add(right);
    
    // Arch (torus)
    const arch = new THREE.TorusGeometry(scale[0] * 0.3, scale[0] * 0.1, 8, 100, 0, Math.PI);
    const archMesh = new THREE.Mesh(arch);
    archMesh.position.y = scale[1] * 0.5;
    archMesh.rotation.z = Math.PI / 2;
    group.add(archMesh);
    
    return group;
  }

  /**
   * Helper: Create cactus geometry
   */
  createCactusGeometry(scale) {
    const group = new THREE.Group();
    
    // Main body
    const body = new THREE.CylinderGeometry(scale[0] * 0.2, scale[0] * 0.2, scale[1], 8);
    const bodyMesh = new THREE.Mesh(body);
    group.add(bodyMesh);
    
    // Spines (small cylinders)
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2;
      const spine = new THREE.CylinderGeometry(scale[0] * 0.05, scale[0] * 0.05, scale[0] * 0.3, 4);
      const spineMesh = new THREE.Mesh(spine);
      spineMesh.position.x = Math.cos(angle) * scale[0] * 0.25;
      spineMesh.position.z = Math.sin(angle) * scale[0] * 0.25;
      spineMesh.rotation.z = Math.PI / 4;
      group.add(spineMesh);
    }
    
    return group;
  }

  /**
   * Helper: Create coral geometry
   */
  createCoralGeometry(scale) {
    const group = new THREE.Group();
    for (let i = 0; i < 3; i++) {
      const branch = new THREE.ConeGeometry(scale[0] * 0.15, scale[1], 8);
      const branchMesh = new THREE.Mesh(branch);
      branchMesh.position.x = (i - 1) * scale[0] * 0.3;
      branchMesh.rotation.z = (Math.random() - 0.5) * 0.5;
      group.add(branchMesh);
    }
    return group;
  }

  /**
   * Helper: Create crystal geometry
   */
  createCrystalGeometry(scale) {
    return new THREE.OctahedronGeometry(scale[0] * 0.3);
  }

  /**
   * Helper: Create mushroom geometry
   */
  createMushroomGeometry(scale) {
    const group = new THREE.Group();
    
    // Stem
    const stem = new THREE.CylinderGeometry(scale[0] * 0.1, scale[0] * 0.1, scale[1] * 0.6, 8);
    const stemMesh = new THREE.Mesh(stem);
    stemMesh.position.y = scale[1] * 0.3;
    group.add(stemMesh);
    
    // Cap
    const cap = new THREE.SphereGeometry(scale[0] * 0.3, 8, 8);
    const capMesh = new THREE.Mesh(cap);
    capMesh.position.y = scale[1] * 0.7;
    capMesh.scale.y = 0.6;
    group.add(capMesh);
    
    return group;
  }

  /**
   * Helper: Create temple geometry
   */
  createTempleGeometry(scale) {
    const group = new THREE.Group();
    
    // Base platform
    const base = new THREE.BoxGeometry(scale[0] * 1.2, scale[1] * 0.2, scale[2] * 1.2);
    const baseMesh = new THREE.Mesh(base);
    group.add(baseMesh);
    
    // Main structure
    const main = new THREE.BoxGeometry(scale[0], scale[1] * 0.8, scale[2]);
    const mainMesh = new THREE.Mesh(main);
    mainMesh.position.y = scale[1] * 0.5;
    group.add(mainMesh);
    
    // Spire
    const spire = new THREE.ConeGeometry(scale[0] * 0.3, scale[1] * 0.5, 8);
    const spireMesh = new THREE.Mesh(spire);
    spireMesh.position.y = scale[1] * 1.2;
    group.add(spireMesh);
    
    return group;
  }

  /**
   * Helper: Create fortress geometry
   */
  createFortressGeometry(scale) {
    const group = new THREE.Group();
    
    // Main walls
    const wall = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const wallMesh = new THREE.Mesh(wall);
    group.add(wallMesh);
    
    // Towers at corners
    const towerRadius = scale[0] * 0.15;
    const cornerPositions = [
      [-scale[0] * 0.4, 0, -scale[2] * 0.4],
      [scale[0] * 0.4, 0, -scale[2] * 0.4],
      [-scale[0] * 0.4, 0, scale[2] * 0.4],
      [scale[0] * 0.4, 0, scale[2] * 0.4]
    ];
    
    for (const pos of cornerPositions) {
      const tower = new THREE.CylinderGeometry(towerRadius, towerRadius, scale[1] * 1.2, 8);
      const towerMesh = new THREE.Mesh(tower);
      towerMesh.position.set(...pos);
      group.add(towerMesh);
    }
    
    return group;
  }

  /**
   * Helper: Create cylinder geometry
   */
  createCylinderGeometry(scale) {
    return new THREE.CylinderGeometry(scale[0] * 0.2, scale[0] * 0.2, scale[1], 32);
  }

  /**
   * Clear the scene
   */
  clearWorld() {
    for (const obj of this.objects) {
      this.scene.remove(obj);
    }
    for (const light of this.lights) {
      this.scene.remove(light);
      if (light.target) {
        this.scene.remove(light.target);
      }
    }
    this.objects = [];
    this.lights = [];
  }
}
