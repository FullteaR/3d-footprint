import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { disposeMaterials, disposeModel } from "./dispose";

// Renders a GLB (terrain + track, with per-body colors) from the backend.
// The mesh is Z-up (millimetres); we tip it to Y-up for natural orbiting.
export function Preview({ glb, errorText }: { glb: ArrayBuffer | null; errorText: string }) {
  const [failed, setFailed] = useState(false);
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const modelRef = useRef<THREE.Object3D>();

  // Mount once: scene, camera, renderer, controls, lights, render loop.
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1721); // matches --well in ui.css
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(0, 150, 150);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setFailed(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(120, 200, 100);
    scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.6);
    dir2.position.set(-100, 120, -80);
    scene.add(dir2);
    // Raking under-light: without it the underside gets only uniform
    // ambient light and the engraved credit stamp is invisible.
    const under = new THREE.DirectionalLight(0xffffff, 0.5);
    under.position.set(200, -60, 50);
    scene.add(under);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;

    const resize = () => {
      const w = mount.clientWidth;
      const h = Math.max(1, mount.clientHeight);
      renderer.setSize(w, h); // updateStyle=true so the canvas fills the box
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      if (modelRef.current) {
        scene.remove(modelRef.current);
        disposeModel(modelRef.current);
        modelRef.current = undefined;
      }
      renderer.dispose();
      renderer.forceContextLoss();
      mount.removeChild(renderer.domElement);
      sceneRef.current = undefined;
      cameraRef.current = undefined;
      controlsRef.current = undefined;
    };
  }, []);

  // Reload model whenever a new GLB arrives.
  useEffect(() => {
    let stale = false;
    if (modelRef.current) {
      sceneRef.current?.remove(modelRef.current);
      disposeModel(modelRef.current);
      modelRef.current = undefined;
    }
    if (!glb || !sceneRef.current) return;
    setFailed(false);
    const scene = sceneRef.current!;
    const loader = new GLTFLoader();
    const onError = () => { if (!stale) setFailed(true); };
    try {
      loader.parse(glb.slice(0), "", (gltf) => {
        const model = gltf.scene;
        if (stale) { disposeModel(model); return; }
        model.rotation.x = -Math.PI / 2; // Z-up (mm) -> Y-up

        // Replace trimesh's PBR/metallic material with a non-metallic one so the
        // per-body colors read true and bright. Smooth shading (flatShading off)
        // uses the crease-smoothed normals the backend baked: terrain shades as a
        // surface while walls/rims/building edges stay crisp. DoubleSide keeps
        // building faces (whose source normals are inconsistent) from going dark.
        const replacedMaterials: THREE.Material[] = [];
        model.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh) return;
          const hasColor = !!mesh.geometry.getAttribute("color");
          replacedMaterials.push(...(Array.isArray(mesh.material) ? mesh.material : [mesh.material]));
          mesh.material = new THREE.MeshStandardMaterial({
            vertexColors: hasColor,
            color: hasColor ? 0xffffff : 0xc2b280,
            metalness: 0,
            roughness: 0.85,
            side: THREE.DoubleSide,
            flatShading: false,
          });
        });
        disposeMaterials(replacedMaterials);

        scene.add(model);
        modelRef.current = model;
        fitCamera(model);
      }, onError);
    } catch {
      onError();
    }
    return () => { stale = true; };
  }, [glb]);

  function fitCamera(model: THREE.Object3D) {
    const camera = cameraRef.current!;
    const controls = controlsRef.current!;
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.6;
    const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
    controls.target.copy(center);
    camera.position.set(center.x, center.y + dist * 0.6, center.z + dist);
    camera.near = dist / 100;
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    controls.update();
  }

  return <div ref={mountRef} style={{ width: "100%", height: "100%", minHeight: 360 }}>
    {failed && <p role="alert">{errorText}</p>}
  </div>;
}
