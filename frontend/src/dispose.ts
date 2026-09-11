import * as THREE from "three";

export function disposeMaterials(materials: THREE.Material[]) {
  const textures = new Set<THREE.Texture>();
  for (const material of new Set(materials)) {
    for (const value of Object.values(material)) {
      if (value instanceof THREE.Texture) textures.add(value);
    }
    material.dispose();
  }
  for (const texture of textures) texture.dispose();
}

export function disposeModel(model: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials: THREE.Material[] = [];
  model.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    geometries.add(mesh.geometry);
    materials.push(...(Array.isArray(mesh.material) ? mesh.material : [mesh.material]));
  });
  for (const geometry of geometries) geometry.dispose();
  disposeMaterials(materials);
}
