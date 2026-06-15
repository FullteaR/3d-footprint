import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Renders a GLB (terrain + track, with per-body colors) from the backend.
// The mesh is Z-up (millimetres); we tip it to Y-up for natural orbiting.
export function Preview({ glb }: { glb: ArrayBuffer | null }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const modelRef = useRef<THREE.Object3D>();

  // Mount once: scene, camera, renderer, controls, lights, render loop.
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(0, 150, 150);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
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

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
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
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // Reload model whenever a new GLB arrives.
  useEffect(() => {
    if (!glb) return;
    const scene = sceneRef.current!;
    const loader = new GLTFLoader();
    loader.parse(glb.slice(0), "", (gltf) => {
      if (modelRef.current) scene.remove(modelRef.current);
      const model = gltf.scene;
      model.rotation.x = -Math.PI / 2; // Z-up (mm) -> Y-up

      // Replace trimesh's PBR/metallic material with a non-metallic one so the
      // per-body colors read true and bright. Smooth shading (flatShading off)
      // uses the crease-smoothed normals the backend baked: terrain shades as a
      // surface while walls/rims/building edges stay crisp. DoubleSide keeps
      // building faces (whose source normals are inconsistent) from going dark.
      model.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const hasColor = !!mesh.geometry.getAttribute("color");
        mesh.material = new THREE.MeshStandardMaterial({
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0xc2b280,
          metalness: 0,
          roughness: 0.85,
          side: THREE.DoubleSide,
          flatShading: false,
        });
      });

      scene.add(model);
      modelRef.current = model;
      fitCamera(model);
    });
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

  return <div ref={mountRef} style={{ width: "100%", height: "100%", minHeight: 360 }} />;
}
