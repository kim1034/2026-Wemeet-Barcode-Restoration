/**
 * 첫 화면 인트로.
 *
 * 송장이 붙은 화물이 지나가는 컨베이어 라인을 멀리서 내려다보다가, 판독 카메라 쪽으로
 * 다가가 렌즈 속으로 들어간다. 렌즈가 보는 장면이 그대로 「실시간 영상」 칸으로
 * 줄어들면서 작업 화면이 뜬다 — "이 카메라가 보는 것이 저 칸이다" 를 한 번에 보여 준다.
 *
 * 판독 기능과는 아무것도 주고받지 않는다. WebGL 을 못 쓰거나 어디서 실패해도
 * onLand() 를 불러 작업 화면은 그대로 뜬다. 언제 돌지는 index.html 머리에서 정한다.
 */
import * as THREE from "/assets/three.module.min.js";

const ORANGE = 0xdf6a2c;
const BG = 0xeceef1; // index.html 의 인트로 가림막과 같은 색 — 넘어갈 때 이음매가 안 보인다

const START = 0.15; // 첫 프레임 뒤 움직이기 시작하기까지(초)
const TRAVEL = 4.7; // 라인 전경 → 렌즈 속
const HOLD = 0.35; // 렌즈 시점에 머문다
const LAND = 0.85; // 화면 전체 → 「실시간 영상」 칸
const FADE = 0.35;

const BELT_TOP = 0.77;
const BELT_LEN = 44;
// 판독 카메라 높이. 렌즈 시점에서 화물 한 개가 통째로 들어오도록 벨트에서 1.6m 넘게 띄운다
const RIG_Y = 2.75;
const GANTRY_H = 3.1;
const LENS_Y = RIG_Y - 0.29;

const clamp01 = (t) => Math.min(1, Math.max(0, t));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const smooth = (a, b, t) => {
  const x = clamp01((t - a) / (b - a));
  return x * x * (3 - 2 * x);
};

// 매번 같은 화물 배치가 나오게 — 무작위면 시연 때마다 장면이 달라진다
function seeded(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// 텍스처 — 그림 파일 없이 캔버스로 그린다
// ---------------------------------------------------------------------------
function canvasTexture(w, h, draw, aniso) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = aniso;
  return tex;
}

function labelTexture(rand, aniso) {
  return canvasTexture(384, 256, (g) => {
    g.fillStyle = "#fbfaf7";
    g.fillRect(0, 0, 384, 256);
    g.fillStyle = "#a3a8ae";
    g.fillRect(24, 20, 120 + rand() * 80, 9);
    g.fillRect(24, 36, 60 + rand() * 60, 7);
    g.fillStyle = "#df6a2c";
    g.fillRect(336, 18, 26, 26);
    g.fillStyle = "#14161a";
    let x = 30;
    const top = 60;
    const h = 126;
    const bar = (w, ink, extra = 0) => {
      if (ink) g.fillRect(x, top, w, h + extra);
      x += w;
    };
    bar(3, true, 12);
    bar(3, false);
    bar(3, true, 12);
    while (x < 336) {
      bar(2 + Math.floor(rand() * 4) * 2, true);
      bar(2 + Math.floor(rand() * 3) * 2, false);
    }
    bar(3, true, 12);
    bar(3, false);
    bar(3, true, 12);
    let d = "880";
    for (let i = 0; i < 10; i++) d += Math.floor(rand() * 10);
    g.font = "600 25px ui-monospace, Menlo, monospace";
    g.fillText(`${d[0]}  ${d.slice(1, 7)}  ${d.slice(7)}`, 34, 232);
  }, aniso);
}

function beltTexture(aniso) {
  const tex = canvasTexture(128, 32, (g) => {
    g.fillStyle = "#2a2e35";
    g.fillRect(0, 0, 128, 32);
    g.fillStyle = "#3b4049";
    g.fillRect(0, 0, 10, 32);
  }, aniso);
  tex.wrapS = THREE.RepeatWrapping;
  tex.repeat.set(BELT_LEN / 0.5, 1); // 0.5m 마다 이음매 한 줄
  return tex;
}

function floorTexture(aniso) {
  const tex = canvasTexture(512, 512, (g) => {
    g.fillStyle = "#e5e8ec";
    g.fillRect(0, 0, 512, 512);
    g.fillStyle = "#d7dce2";
    for (let i = 0; i < 512; i += 64) {
      g.fillRect(i, 0, 1.5, 512);
      g.fillRect(0, i, 512, 1.5);
    }
  }, aniso);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(15, 15); // 1m 격자
  return tex;
}

// ---------------------------------------------------------------------------
// 장면
// ---------------------------------------------------------------------------
function buildWorld(renderer) {
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const rand = seeded(20261116);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(BG, 14, 40);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xcfd5dc, 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, 2.3);
  sun.position.set(-6, 12, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 12, bottom: -12, near: 1, far: 40 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  scene.add(sun);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(120, 120),
    new THREE.MeshStandardMaterial({ map: floorTexture(aniso), roughness: 0.95 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // 먼 기둥 — 공간의 크기를 잡아 준다. 안개 속으로 사라진다
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0xdde1e6, roughness: 0.9 });
  for (let x = -20; x <= 20; x += 6.5) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.5, 7, 0.5), pillarMat);
    p.position.set(x, 3.5, -11.5);
    scene.add(p);
  }

  const frameMat = new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.6, metalness: 0.3 });
  const legMat = new THREE.MeshStandardMaterial({ color: 0x3a3f47, roughness: 0.7 });
  const railMat = new THREE.MeshStandardMaterial({ color: 0xc9ced6, roughness: 0.35, metalness: 0.6 });
  const glowMat = new THREE.MeshBasicMaterial({ color: ORANGE });
  const cardboard = [0xc89b6d, 0xb98a5c, 0xd4ab7f, 0xa97d52, 0xcfa476].map(
    (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.85 }),
  );
  const labels = Array.from({ length: 8 }, () =>
    new THREE.MeshBasicMaterial({ map: labelTexture(rand, aniso), toneMapped: false }),
  );

  const belts = [];
  const boxes = [];

  function belt(z, speed, main) {
    const tex = beltTexture(aniso);
    const top = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.75 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.22, 1.7), frameMat);
    body.position.set(0, 0.62, z);
    body.castShadow = body.receiveShadow = true;
    scene.add(body);
    const surface = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.04, 1.5), [frameMat, frameMat, top, frameMat, frameMat, frameMat]);
    surface.position.set(0, BELT_TOP - 0.02, z);
    surface.receiveShadow = true;
    scene.add(surface);
    for (const side of [-1, 1]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.1, 0.06), railMat);
      rail.position.set(0, 0.82, z + side * 0.8);
      rail.castShadow = true;
      scene.add(rail);
      if (main) {
        // 우리 라인만 주황 띠 — 멀리서도 어느 라인 이야기인지 보인다
        const strip = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.012, 0.022), glowMat);
        strip.position.set(0, 0.876, z + side * 0.8);
        scene.add(strip);
      }
    }
    for (let x = -BELT_LEN / 2 + 1; x < BELT_LEN / 2; x += 3) {
      for (const side of [-1, 1]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.51, 0.08), legMat);
        leg.position.set(x, 0.255, z + side * 0.7);
        leg.castShadow = true;
        scene.add(leg);
      }
    }
    belts.push({ tex, speed });

    let x = -BELT_LEN / 2 + rand() * 2;
    while (x < BELT_LEN / 2) {
      const w = 0.5 + rand() * 0.4;
      const h = 0.3 + rand() * 0.3;
      const d = 0.5 + rand() * 0.4;
      const box = new THREE.Group();
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), cardboard[Math.floor(rand() * cardboard.length)]);
      mesh.position.y = h / 2;
      mesh.castShadow = mesh.receiveShadow = true;
      box.add(mesh);
      const lw = Math.min(w, d) * 0.62;
      const label = new THREE.Mesh(new THREE.PlaneGeometry(lw, lw / 1.5), labels[Math.floor(rand() * labels.length)]);
      label.rotation.x = -Math.PI / 2;
      // 일부 송장은 비뚤게 붙어 있다 — 우리가 풀려는 문제가 이것이다
      label.rotation.z = rand() < 0.35 ? (rand() - 0.5) * 0.9 : (rand() - 0.5) * 0.12;
      label.position.y = h + 0.003;
      box.add(label);
      box.position.set(x, BELT_TOP, z + (rand() - 0.5) * 0.3);
      scene.add(box);
      boxes.push({ obj: box, speed });
      x += w + 1.2 + rand() * 1.4;
    }
  }

  belt(0, 1.25, true);
  belt(-3.6, 1.05, false);
  belt(3.6, 1.45, false);
  belt(-7.2, 1.15, false);

  // 옆 라인의 기존 스캐너 — 회색, 우리 것과 구분
  for (const z of [-3.6, 3.6]) {
    const arch = new THREE.Group();
    for (const side of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.1, 2.0, 0.1), railMat);
      post.position.set(0, 1.0, side * 1.05);
      arch.add(post);
    }
    const beam = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.14, 2.2), railMat);
    beam.position.y = 2.0;
    arch.add(beam);
    arch.position.set(z > 0 ? 5 : -5, 0, z);
    arch.traverse((o) => (o.castShadow = true));
    scene.add(arch);
  }

  // 우리 판독 카메라 — x=0, 가운데 라인 위
  const metal = new THREE.MeshStandardMaterial({ color: 0xd5d9df, roughness: 0.3, metalness: 0.7 });
  const gantry = new THREE.Group();
  for (const side of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.12, GANTRY_H, 0.12), metal);
    post.position.set(0, GANTRY_H / 2, side * 1.15);
    gantry.add(post);
  }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.16, 2.5), metal);
  beam.position.y = GANTRY_H;
  gantry.add(beam);
  gantry.traverse((o) => (o.castShadow = true));
  scene.add(gantry);

  const rig = new THREE.Group();
  rig.position.set(0, RIG_Y, 0);
  const housing = new THREE.Mesh(
    new THREE.BoxGeometry(0.42, 0.26, 0.3),
    new THREE.MeshStandardMaterial({ color: 0x1e2228, roughness: 0.4, metalness: 0.4 }),
  );
  housing.castShadow = true;
  rig.add(housing);
  const mountLen = GANTRY_H - 0.08 - (RIG_Y + 0.13);
  const mount = new THREE.Mesh(new THREE.BoxGeometry(0.06, mountLen, 0.06), metal);
  mount.position.y = 0.13 + mountLen / 2;
  rig.add(mount);
  const barrel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.075, 0.085, 0.16, 40),
    new THREE.MeshStandardMaterial({ color: 0x15181d, roughness: 0.35, metalness: 0.5 }),
  );
  barrel.position.y = -0.21;
  rig.add(barrel);
  const glass = new THREE.Mesh(
    new THREE.CircleGeometry(0.066, 40),
    new THREE.MeshStandardMaterial({ color: 0x0b1a2a, roughness: 0.05, metalness: 0.2, emissive: 0x0a2a44, emissiveIntensity: 0.6 }),
  );
  glass.rotation.x = Math.PI / 2;
  glass.position.y = -0.291;
  rig.add(glass);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.083, 0.012, 12, 48), glowMat);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -0.285;
  rig.add(ring);
  const ledMat = new THREE.MeshBasicMaterial({ color: ORANGE });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), ledMat);
  led.position.set(0.15, 0.06, 0.152);
  rig.add(led);
  // 카메라 몸통의 [PIMM.] 명판 — 다가가면서 읽힌다
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(0.26, 0.065),
    new THREE.MeshBasicMaterial({
      map: canvasTexture(512, 128, (g) => {
        g.fillStyle = "#1e2228";
        g.fillRect(0, 0, 512, 128);
        g.strokeStyle = "#df6a2c";
        g.lineWidth = 9;
        const b = 26;
        g.beginPath();
        g.moveTo(40 + b, 18); g.lineTo(40, 18); g.lineTo(40, 110); g.lineTo(40 + b, 110);
        g.moveTo(472 - b, 18); g.lineTo(472, 18); g.lineTo(472, 110); g.lineTo(472 - b, 110);
        g.stroke();
        g.font = "800 72px system-ui, sans-serif";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillStyle = "#f4f4f2";
        g.fillText("PIMM.", 256, 68);
      }, aniso),
      toneMapped: false,
    }),
  );
  plate.position.set(-0.04, 0.0, 0.151);
  rig.add(plate);
  scene.add(rig);

  // 판독 영역 — 렌즈에서 벨트까지 퍼지는 주황 사각뿔
  const coneH = LENS_Y - BELT_TOP;
  const halfZ = coneH * Math.tan((30 * Math.PI) / 180);
  const halfX = halfZ * 1.6;
  const coneGeo = new THREE.ConeGeometry(halfX * Math.SQRT2, coneH, 4, 1, true);
  coneGeo.rotateY(Math.PI / 4);
  const coneMat = new THREE.MeshBasicMaterial({
    color: ORANGE, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false,
  });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  cone.scale.set(1, 1, halfZ / halfX);
  cone.position.set(0, BELT_TOP + coneH / 2, 0);
  scene.add(cone);

  // 벨트 위를 오가는 판독선
  const laser = new THREE.Group();
  const line = new THREE.Mesh(new THREE.PlaneGeometry(0.018, 1.3), new THREE.MeshBasicMaterial({ color: ORANGE, toneMapped: false }));
  const halo = new THREE.Mesh(
    new THREE.PlaneGeometry(0.16, 1.3),
    new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0.22, depthWrite: false, toneMapped: false }),
  );
  for (const m of [line, halo]) {
    m.rotation.x = -Math.PI / 2;
    laser.add(m);
  }
  laser.position.y = BELT_TOP + 0.012;
  scene.add(laser);

  // 판독 카메라 바로 아래를 지나는 화물을 따뜻하게 비춘다
  const spot = new THREE.PointLight(ORANGE, 3.2, 3.2, 2);
  spot.position.set(0, LENS_Y - 0.1, 0);
  scene.add(spot);

  let clock = 0;
  return {
    scene,
    update(dt) {
      clock += dt;
      for (const b of belts) b.tex.offset.x -= (b.speed * dt) / 0.5;
      for (const b of boxes) {
        b.obj.position.x += b.speed * dt;
        if (b.obj.position.x > BELT_LEN / 2) b.obj.position.x -= BELT_LEN;
      }
      laser.position.x = Math.sin(clock * 2.4) * 0.62;
      ledMat.color.setHex(Math.sin(clock * 7) > 0 ? ORANGE : 0x7a3a18);
    },
    // 렌즈 속으로 들어갈 때 카메라 몸통과 사각뿔을 걷는다 — 안쪽 면이 화면을 가린다
    approach(u) {
      coneMat.opacity = 0.08 * (1 - smooth(0.86, 0.97, u));
      rig.visible = u < 0.975;
    },
  };
}

// 카메라가 지나는 길. 위치와 시선을 같은 매개변수로 묶어 움직인다
const PATH = new THREE.CatmullRomCurve3([
  new THREE.Vector3(-13, 6.4, 10.5),
  new THREE.Vector3(-7.5, 5.0, 6.4),
  new THREE.Vector3(-3.2, 3.6, 3.0),
  new THREE.Vector3(-1.0, RIG_Y + 0.2, 1.05),
  new THREE.Vector3(-0.15, RIG_Y - 0.25, 0.25),
  new THREE.Vector3(0, LENS_Y - 0.04, 0.02),
]);
const GAZE = new THREE.CatmullRomCurve3([
  new THREE.Vector3(1.5, 0.9, -1.2),
  new THREE.Vector3(0.6, 1.3, -0.4),
  new THREE.Vector3(0, RIG_Y - 0.4, 0),
  new THREE.Vector3(0, RIG_Y - 0.15, 0),
  new THREE.Vector3(0, 1.2, -0.05),
  new THREE.Vector3(0, BELT_TOP, 0),
]);

const CSS = `
.pimm-intro { position: fixed; z-index: 1000; left: 0; top: 0; width: 100vw; height: 100vh; overflow: hidden;
  background: #eceef1; cursor: pointer; will-change: left, top, width, height, opacity; }
.pimm-intro canvas { position: absolute; display: block; }
.pi-veil { position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(20deg, rgba(236,238,241,.92) 0%, rgba(236,238,241,.55) 32%, rgba(236,238,241,0) 58%);
  opacity: 0; transition: opacity .8s ease; }
.pi-title { position: absolute; left: clamp(24px, 6vw, 88px); bottom: clamp(28px, 10vh, 104px); color: #16181d;
  opacity: 0; transform: translateY(14px); transition: opacity .7s ease, transform .9s cubic-bezier(.2,.8,.2,1); pointer-events: none; }
.pimm-intro.titled .pi-title, .pimm-intro.titled .pi-veil { opacity: 1; transform: none; }
.pi-mark { display: inline-block; position: relative; padding: 6px 14px; font: 800 22px/1 "IBM Plex Sans KR", system-ui, sans-serif; letter-spacing: -.04em; }
.pi-mark::before, .pi-mark::after { content: ""; position: absolute; top: 0; bottom: 0; width: 8px; border: 2.5px solid #df6a2c; }
.pi-mark::before { left: 0; border-right: 0; }
.pi-mark::after { right: 0; border-left: 0; }
.pi-mark em, .pi-mark b { font-style: normal; color: #df6a2c; }
.pi-title h2 { margin: 18px 0 10px; font: 700 clamp(26px, 3.1vw, 44px)/1.24 "IBM Plex Sans KR", system-ui, sans-serif; letter-spacing: -.035em; }
.pi-title p { margin: 0; font: 500 14px/1.4 "IBM Plex Sans KR", system-ui, sans-serif; color: #69707a; }
.pi-finder { position: absolute; inset: 10% 8%; opacity: 0; transition: opacity .5s ease; pointer-events: none; }
.pimm-intro.pov .pi-finder { opacity: 1; }
.pi-finder i { position: absolute; width: 34px; height: 34px; border: 2.5px solid #df6a2c; }
.pi-finder i:nth-child(1) { left: 0; top: 0; border-right: 0; border-bottom: 0; }
.pi-finder i:nth-child(2) { right: 0; top: 0; border-left: 0; border-bottom: 0; }
.pi-finder i:nth-child(3) { left: 0; bottom: 0; border-right: 0; border-top: 0; }
.pi-finder i:nth-child(4) { right: 0; bottom: 0; border-left: 0; border-top: 0; }
.pi-cam { position: absolute; left: 4px; top: -30px; display: flex; align-items: center; gap: 8px;
  font: 600 12px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: .1em; color: #fff; text-shadow: 0 1px 3px rgba(0,0,0,.55); }
.pi-cam::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #df6a2c; box-shadow: 0 0 0 4px rgba(223,106,44,.32); }
.pi-scan { position: absolute; left: 0; right: 0; top: 0; height: 2px; opacity: 0; pointer-events: none;
  background: linear-gradient(90deg, transparent, #df6a2c 18%, #ffb07f 50%, #df6a2c 82%, transparent);
  box-shadow: 0 0 22px 5px rgba(223,106,44,.45); }
.pimm-intro.flash .pi-scan { animation: pi-scan .7s cubic-bezier(.4,0,.2,1) forwards; }
@keyframes pi-scan { 0% { top: 0; opacity: 0; } 12% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
.pi-skip { position: absolute; right: 24px; bottom: 24px; appearance: none; cursor: pointer;
  display: inline-flex; align-items: center; gap: 8px; padding: 9px 16px; border-radius: 999px;
  font: 600 13px/1 "IBM Plex Sans KR", system-ui, sans-serif; color: #16181d;
  background: rgba(255,255,255,.6); border: 1px solid rgba(255,255,255,.85);
  -webkit-backdrop-filter: blur(16px) saturate(160%); backdrop-filter: blur(16px) saturate(160%);
  box-shadow: 0 8px 24px -10px rgba(20,28,40,.35); transition: opacity .3s; }
.pi-skip kbd { font: 500 11px/1 "IBM Plex Mono", monospace; color: #69707a; padding: 3px 5px; border-radius: 5px; background: rgba(20,28,40,.07); }
.pimm-intro.landing .pi-skip, .pimm-intro.landing .pi-title, .pimm-intro.landing .pi-veil { opacity: 0; }
`;

/**
 * @param {{ target: HTMLElement | null, onLand: () => void }} opts
 *   target — 장면이 줄어들어 내려앉을 칸(「실시간 영상」 의 영상칸)
 *   onLand — 작업 화면을 띄울 때 부른다. 실패해도 반드시 한 번은 불린다.
 */
export function runIntro({ target, onLand }) {
  let landed = false;
  const land = () => {
    if (landed) return;
    landed = true;
    onLand();
  };

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch {
    land();
    return Promise.resolve(false);
  }

  return new Promise((resolve) => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.className = "pimm-intro";
    root.innerHTML =
      '<div class="pi-veil" aria-hidden="true"></div>' +
      '<div class="pi-finder" aria-hidden="true"><i></i><i></i><i></i><i></i><span class="pi-cam">CAM-01 · SCAN</span></div>' +
      '<div class="pi-scan" aria-hidden="true"></div>' +
      '<div class="pi-title" aria-hidden="true"><div class="pi-mark">P<em>I</em>MM<b>.</b></div>' +
      "<h2>못 읽던 송장 바코드,<br>지금 있는 카메라로 다시 읽습니다</h2><p>노리드 판독기</p></div>" +
      '<button type="button" class="pi-skip">건너뛰기 <kbd>Esc</kbd></button>';
    root.prepend(renderer.domElement);
    document.body.appendChild(root);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setClearColor(BG, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.05;

    const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
    let bufW = 0;
    let bufH = 0;
    let morphing = false;
    let skipping = false;
    let finished = false;
    let raf = 0;
    let t = 0;
    let last = performance.now();
    let flashed = false;

    let world;
    try {
      world = buildWorld(renderer);
    } catch {
      finish();
      return;
    }

    function resize() {
      bufW = window.innerWidth;
      bufH = window.innerHeight;
      renderer.setSize(bufW, bufH, false);
      camera.aspect = bufW / bufH;
      camera.updateProjectionMatrix();
      if (!morphing) place(bufW, bufH);
    }
    // 그리는 크기는 화면 전체로 두고, 줄어드는 동안은 칸을 꽉 채우도록(cover) 놓는다 — 찌그러지지 않는다
    function place(w, h) {
      const ar = bufW / bufH;
      let cw = w;
      let ch = w / ar;
      if (ch < h) {
        ch = h;
        cw = h * ar;
      }
      const s = renderer.domElement.style;
      s.width = cw + "px";
      s.height = ch + "px";
      s.left = (w - cw) / 2 + "px";
      s.top = (h - ch) / 2 + "px";
    }

    function morph(k) {
      const r = target && target.getBoundingClientRect();
      if (!r || r.width < 4 || r.height < 4) return;
      const w = lerp(window.innerWidth, r.width, k);
      const h = lerp(window.innerHeight, r.height, k);
      const s = root.style;
      s.left = lerp(0, r.left, k) + "px";
      s.top = lerp(0, r.top, k) + "px";
      s.width = w + "px";
      s.height = h + "px";
      s.borderRadius = 16 * k + "px";
      s.boxShadow = `0 ${24 * k}px ${60 * k}px -${20 * k}px rgba(20,28,40,${0.35 * k})`;
      place(w, h);
    }

    // 한 프레임이라도 던지면 덮개가 작업 화면을 영영 가린다 — 무엇이든 실패하면 접고 나간다
    function frame(now) {
      try {
        step(now);
      } catch {
        finish();
      }
    }

    function step(now) {
      // 느린 기계에서도 실제 시간대로 흐르게 0.1초까지는 그대로 받는다(그 이상은 탭이 숨었던 것)
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      t += dt;
      world.update(dt);

      const u = easeInOutCubic(clamp01((t - START) / TRAVEL));
      const k = PATH.getUtoTmapping(u);
      camera.position.copy(PATH.getPoint(k));
      camera.lookAt(GAZE.getPoint(k));
      camera.fov = 40 + 20 * smooth(0.82, 1, u);
      camera.updateProjectionMatrix();
      world.approach(u);

      root.classList.toggle("titled", t > 0.35 && t < 2.6);
      if (u > 0.9) root.classList.add("pov");
      if (!flashed && t > START + TRAVEL - 0.15) {
        flashed = true;
        root.classList.add("flash");
      }

      const landT = t - (START + TRAVEL + HOLD);
      if (landT >= 0 && !skipping) {
        if (!morphing) {
          morphing = true;
          root.classList.add("landing");
          land();
        }
        morph(easeInOutCubic(clamp01(landT / LAND)));
        if (landT > LAND) root.style.opacity = String(1 - clamp01((landT - LAND) / FADE));
        if (landT >= LAND + FADE) {
          finish();
          return;
        }
      }

      renderer.render(world.scene, camera);
      raf = requestAnimationFrame(frame);
    }

    function skip() {
      if (skipping || finished) return;
      skipping = true;
      land();
      root.style.transition = "opacity .3s ease";
      root.style.opacity = "0";
      setTimeout(finish, 320);
    }
    function onKey(e) {
      if (e.key === "Escape" || e.key === "Enter" || e.key === " ") skip();
    }

    function finish() {
      if (finished) return;
      finished = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", onKey);
      land();
      // 덮개부터 걷는다. 아래 정리가 실패해도 작업 화면은 가리지 않는다
      root.remove();
      style.remove();
      // GPU 메모리를 돌려준다 — 작업 화면은 몇 시간씩 열려 있다
      try {
        world?.scene.traverse((o) => {
          o.geometry?.dispose();
          for (const m of [].concat(o.material || [])) {
            m.map?.dispose();
            m.dispose();
          }
        });
        renderer.dispose();
        renderer.forceContextLoss();
      } catch {
        // 정리 실패는 화면에 영향이 없다
      }
      resolve(true);
    }

    try {
      resize();
      window.addEventListener("resize", resize);
      window.addEventListener("keydown", onKey);
      root.addEventListener("click", skip);
      root.querySelector(".pi-skip").focus({ preventScroll: true });
      // 셰이더를 먼저 굽는다 — 첫 프레임에 굽느라 멈칫하면 시작이 끊겨 보인다
      camera.position.copy(PATH.getPoint(0));
      camera.lookAt(GAZE.getPoint(0));
      renderer.compile(world.scene, camera);
    } catch {
      finish();
      return;
    }
    raf = requestAnimationFrame((now) => {
      last = now;
      frame(now);
    });
  });
}
