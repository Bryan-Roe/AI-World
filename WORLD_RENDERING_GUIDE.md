# World Rendering Guide

## Overview
The world rendering system enables AI-generated worlds to be displayed in 3D using Three.js and the WorldRenderer class.

## How It Works

### 1. **World Generation** (`world_generator.html`)
- User configures world settings (biome, terrain type, custom prompt)
- Clicks "Generate World" button
- Server calls `/api/generate-world` endpoint
- AI model generates JSON world description
- World data is saved to browser localStorage
- User clicks "View in 3D" button

### 2. **Data Flow**
```
world_generator.html
    ↓
/api/generate-world (server.js)
    ↓
AI Model (Python world_generator_train.py)
    ↓
World JSON returned
    ↓
localStorage.setItem('generatedWorld', JSON.stringify(worldData))
    ↓
Redirect to game.html?loadGenerated=true
```

### 3. **World Rendering** (`game.js` + `WorldRenderer.js`)

#### Initialization Flow
1. `game.js` imports `WorldRenderer` class from `WorldRenderer.js`
2. WorldRenderer instance created: `this.worldRenderer = new WorldRenderer(scene, camera, renderer)`
3. `checkForGeneratedWorld()` method checks URL parameters
4. If `loadGenerated=true`, retrieves world from localStorage
5. Calls `loadGeneratedWorld(worldData)` with the generated world data

#### Rendering Pipeline
```javascript
// In game.js init() method:
this.worldRenderer = new WorldRenderer(this.scene, this.camera, this.renderer);
this.checkForGeneratedWorld();

// In checkForGeneratedWorld():
const savedWorld = localStorage.getItem('generatedWorld');
this.loadGeneratedWorld(JSON.parse(savedWorld));

// In loadGeneratedWorld():
const result = await this.worldRenderer.renderWorld(worldData);
// Updates camera and displays status
```

#### WorldRenderer Methods

**renderWorld(worldData)**
- Main entry point for rendering an AI-generated world
- Clears previous world
- Sets atmosphere (fog, sky, lighting)
- Creates terrain objects
- Creates structures (buildings, monuments, etc.)
- Creates vegetation (trees, bushes, flowers, etc.)
- Creates lights (ambient, directional, point, spot)
- Adds weather effects
- Returns object with counts (objectCount, lightCount, name)

**Individual Creation Methods**
- `setAtmosphere(worldData)` - Sets fog, sky color, ambient lighting
- `createTerrain(terrainObjects)` - Creates ground and terrain features
- `createStructures(structures)` - Creates buildings, towers, bridges, etc.
- `createVegetation(vegetation)` - Creates trees, bushes, plants, etc.
- `createLights(lightsData)` - Creates various light types
- `addWeatherEffects(weatherType)` - Adds environmental effects

**Helper Methods for Geometry Creation**
- `createTreeGeometry()` - Creates tree with trunk and canopy
- `createHouseGeometry()` - Creates simple house with roof
- `createTowerGeometry()` - Creates cylindrical tower
- `createTempleGeometry()` - Creates temple with base, structure, and spire
- `createFortressGeometry()` - Creates fortified structure with corner towers
- And many more specialized geometry creators...

### 4. **Scene Clearing** 

The `clearSceneObjects()` method:
- Removes all objects from the scene
- Disposes geometry and materials
- Clears chunk data if in infinite mode
- Prepares scene for new world rendering

### 5. **World Data Structure** (JSON Format)

Expected structure of generated world JSON:
```javascript
{
  "name": "World Name",
  "sky_color": "0x87ceeb",  // Hex color
  "fog_color": "0x87ceeb",
  "fog_density": 0.005,
  "terrain_objects": [
    {
      "type": "plane|cylinder|sphere",
      "position": [x, y, z],
      "scale": [width, height, depth],
      "color": "0xffaa00",
      "roughness": 0.7,
      "metalness": 0.1,
      "receives_shadow": true
    }
  ],
  "structures": [
    {
      "type": "cube|sphere|house|tower|temple|fortress|etc",
      "position": [x, y, z],
      "scale": [w, h, d],
      "color": "0xff0000",
      "rotation": 45,  // degrees
      "roughness": 0.5,
      "metalness": 0.2,
      "emissive": "0x000000",
      "casts_shadow": true
    }
  ],
  "vegetation": [
    {
      "type": "tree|bush|grass|flower|cactus|coral|crystal|mushroom",
      "position": [x, y, z],
      "scale": [w, h, d],
      "color": "0x228b22",
      "animation": "sway|pulse|none"
    }
  ],
  "lights": [
    {
      "type": "ambient|directional|point|spot|hemisphere",
      "color": "0xffffff",
      "intensity": 1.0,
      "position": [x, y, z],  // Optional
      "direction": [x, y, z],  // Optional
      "distance": 100,  // For point/spot lights
      "decay": 2,  // For point lights
      "angle": 1.0,  // For spot lights
      "penumbra": 0.5  // For spot lights
    }
  ],
  "atmosphere": {
    "ambientLight": 0.5
  },
  "weather": "sunny|rainy|stormy|foggy|snowy"
}
```

## Usage Example

### For Developers

```javascript
// In game.js
const worldData = {
  name: "Mystical Forest",
  sky_color: "0x87ceeb",
  terrain_objects: [...],
  structures: [...],
  vegetation: [...],
  lights: [...]
};

// Manually render a world
await game.loadGeneratedWorld(worldData);
```

### For End Users

1. Open `http://localhost:3000/world_generator.html`
2. Select biome (Forest, Desert, Ocean, etc.)
3. Select terrain type (Flat, Hilly, Mountainous, etc.)
4. Optionally enter custom prompt
5. Click "Generate World" button
6. Wait for AI to generate the world
7. View stats (terrain features, structures, lights)
8. Click "View in 3D" button
9. See the rendered world in the 3D viewer

## Features

### Terrain Rendering
- Planes for flat ground
- Cylinders for mountains/hills
- Spheres for planets/domes
- Custom roughness and metalness

### Structure Variety
- Basic shapes (cube, sphere, cylinder, cone, torus, pyramid)
- Complex structures (house, tower, bridge, arch, pillar, statue, temple, fortress)
- Custom colors, materials, and shadows
- Emissive materials for glowing objects

### Vegetation System
- Trees with trunk and canopy
- Bushes and grass
- Flowers with petals
- Cacti with spines
- Coral with branches
- Crystals
- Mushrooms
- Animation support (sway, pulse)

### Lighting System
- Ambient lights for overall illumination
- Directional lights for sun/moon
- Point lights for local illumination
- Spot lights for focused illumination
- Hemisphere lights for outdoor scenes

### Performance Optimization
- Shadow budget limiting
- Proper material disposal
- Geometry reuse for common shapes
- Efficient scene management

## Testing

### Manual Testing
1. Generate a world with world_generator.html
2. Click "View in 3D"
3. Verify:
   - World loads without errors
   - All objects render correctly
   - Lighting looks appropriate
   - Camera positions well to view the world
   - No performance issues

### Automated Testing
```bash
# Check for syntax errors
npm run check:js

# Run tests
npm test
```

## Troubleshooting

### World Not Rendering
1. Check browser console for errors (F12)
2. Verify localStorage has 'generatedWorld' key
3. Check that WorldRenderer.js is properly imported
4. Ensure Three.js is loaded before game.js

### Poor Performance
1. Reduce structure/vegetation count in generated world
2. Disable shadows in renderer config
3. Reduce fog density
4. Optimize lighting (fewer shadow-casting lights)

### Missing Textures/Materials
1. Verify color hex values are correct
2. Check material parameters (roughness, metalness)
3. Ensure colors are valid 0xRRGGBB format

## Future Enhancements

- [ ] Procedural terrain generation (heightmaps)
- [ ] Texture mapping
- [ ] Physics bodies for interaction
- [ ] Dynamic weather effects (rain particles, snow, etc.)
- [ ] Water simulation
- [ ] Path finding for AI navigation
- [ ] Post-processing effects specific to biomes
- [ ] World persistence/saving
- [ ] Multiplayer world rendering

## Files Involved

- **world_generator.html** - UI for configuring and generating worlds
- **world_generator_train.py** - AI training script that generates worlds
- **server.js** - API endpoint `/api/generate-world`
- **game.js** - Main game logic, handles world loading
- **WorldRenderer.js** - THREE.js-based world renderer
- **game.html** - Main game container

## API Reference

### POST /api/generate-world

Generate a new world based on prompt.

**Request:**
```json
{
  "prompt": "Generate a mystical forest with glowing trees and floating islands"
}
```

**Response:**
```json
{
  "world": {
    "name": "Mystical Forest",
    "terrain_objects": [...],
    "structures": [...],
    "vegetation": [...],
    "lights": [...]
  }
}
```

**Error Response:**
```json
{
  "error": "Generation failed",
  "details": "Model not trained",
  "message": "Please run: python world_generator_train.py"
}
```
