/**
 * 첫 화면 인트로 (5초).
 *
 * 어두운 물류장에서 주황 비닐 택배(송장 바코드가 붙은)가 지나가는 컨베이어 라인을
 * 내려다보다가, 판독 카메라 쪽으로 다가가 렌즈 속으로 들어간다. 렌즈가 보는 장면이
 * 그대로 무대 안의 실제 영상 자리(16:9)로 줄어들면서 작업 화면이 뜬다 — "이 카메라가
 * 보는 것이 저기에 나온다" 를 한 번에 보여 준다.
 *
 * 판독 기능과는 아무것도 주고받지 않는다. WebGL 을 못 쓰거나 어디서 실패해도
 * onLand() 를 불러 작업 화면은 그대로 뜬다. 언제 돌지는 index.html 머리에서 정한다.
 */
import * as THREE from "/assets/three.module.min.js";

const ORANGE = 0xdf6a2c;
const BG = 0x0c0e11; // index.html 의 인트로 가림막과 같은 색 — 넘어갈 때 이음매가 안 보인다

// 시간표(초) — 합계 5초
const START = 0.1; // 첫 프레임 뒤 움직이기 시작하기까지
const TRAVEL = 3.65; // 라인 전경 → 렌즈 속
const HOLD = 0.2; // 렌즈 시점에 머문다
const LAND = 0.75; // 화면 전체 → 「실시간 영상」 무대
const FADE = 0.3;

const BELT_TOP = 0.82;
const BELT_LEN = 44;
// 판독 카메라 높이. 렌즈 시점에서 택배 하나가 통째로 들어오도록 벨트에서 1.6m 넘게 띄운다
const RIG_Y = 2.8;
const GANTRY_H = 3.15;
const LENS_Y = RIG_Y - 0.29;

const clamp01 = (t) => Math.min(1, Math.max(0, t));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOutSine = (t) => -(Math.cos(Math.PI * t) - 1) / 2;
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const smooth = (a, b, t) => {
  const x = clamp01((t - a) / (b - a));
  return x * x * (3 - 2 * x);
};

// 매번 같은 배치가 나오게 — 무작위면 시연 때마다 장면이 달라진다
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
function canvasTexture(w, h, draw, aniso, srgb = true) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"));
  const tex = new THREE.CanvasTexture(c);
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
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
    g.font = "600 25px Pretendard, system-ui, sans-serif";
    g.fillText(`${d[0]}  ${d.slice(1, 7)}  ${d.slice(7)}`, 34, 232);
  }, aniso);
}

function rubberTexture(aniso) {
  const rand = seeded(11);
  const tex = canvasTexture(256, 64, (g) => {
    g.fillStyle = "#1b1d21";
    g.fillRect(0, 0, 256, 64);
    for (let i = 0; i < 1400; i++) {
      const v = 30 + Math.floor(rand() * 18);
      g.fillStyle = `rgba(${v},${v + 2},${v + 5},0.5)`;
      g.fillRect(rand() * 256, rand() * 64, 1, 1);
    }
    g.fillStyle = "#2b2e34"; // 0.6m 마다 벨트 이음매
    g.fillRect(0, 0, 4, 64);
  }, aniso);
  tex.wrapS = THREE.RepeatWrapping;
  tex.repeat.set(BELT_LEN / 0.6, 1);
  return tex;
}

function floorTexture(aniso) {
  const rand = seeded(7);
  const tex = canvasTexture(1024, 1024, (g) => {
    g.fillStyle = "#17191d";
    g.fillRect(0, 0, 1024, 1024);
    for (let i = 0; i < 14000; i++) {
      const v = 24 + Math.floor(rand() * 16);
      g.fillStyle = `rgba(${v},${v + 2},${v + 6},${0.25 + rand() * 0.35})`;
      g.fillRect(rand() * 1024, rand() * 1024, 1 + rand() * 2, 1 + rand() * 2);
    }
    g.fillStyle = "#101114"; // 1m 이음매
    for (let i = 0; i < 1024; i += 128) {
      g.fillRect(i, 0, 2, 1024);
      g.fillRect(0, i, 1024, 2);
    }
  }, aniso);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(15, 15);
  return tex;
}

function dashTexture(aniso) {
  const tex = canvasTexture(128, 8, (g) => {
    g.clearRect(0, 0, 128, 8);
    g.fillStyle = "#ffffff";
    g.fillRect(0, 0, 76, 8);
  }, aniso);
  tex.wrapS = THREE.RepeatWrapping;
  tex.repeat.set(BELT_LEN / 1.2, 1);
  return tex;
}

// 사진관 같은 반사 환경 — 천장 조명 줄이 비닐과 금속에 길게 맺힌다
function studioEnvironment(renderer) {
  const env = new THREE.Scene();
  const room = new THREE.Mesh(
    new THREE.BoxGeometry(30, 14, 30),
    new THREE.MeshBasicMaterial({ color: 0x0b0c0e, side: THREE.BackSide }),
  );
  room.position.y = 5;
  env.add(room);
  const panel = (w, h, color, k, pos, rot) => {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(k), side: THREE.DoubleSide }),
    );
    m.position.set(...pos);
    m.rotation.set(...rot);
    env.add(m);
  };
  for (let i = -2; i <= 2; i++) panel(16, 0.8, 0xffffff, 5, [0, 11.5, i * 4], [Math.PI / 2, 0, 0]);
  panel(10, 3, ORANGE, 1.6, [-14.5, 3, 0], [0, Math.PI / 2, 0]);
  panel(10, 3, 0x8fb2ff, 0.7, [14.5, 4, 0], [0, -Math.PI / 2, 0]);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(env, 0.04).texture;
  pmrem.dispose();
  env.traverse((o) => {
    o.geometry?.dispose();
    o.material?.dispose();
  });
  return tex;
}

// 비닐 택배 — 가운데가 부풀고 테두리가 눌린 베개 모양, 위면에 잔주름
function bagGeometry(w, h, d, rand) {
  const g = new THREE.BoxGeometry(1, 1, 1, 20, 3, 16);
  const pos = g.attributes.position;
  const v = new THREE.Vector3();
  const p1 = rand() * 6.28;
  const p2 = rand() * 6.28;
  const p3 = rand() * 6.28;
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const nx = v.x * 2;
    const nz = v.z * 2;
    const puff = Math.sqrt(Math.max(0, (1 - Math.abs(nx) ** 5) * (1 - Math.abs(nz) ** 5)));
    let y = (v.y + 0.5) * (0.12 + 0.88 * puff); // 바닥 0, 가운데 꼭대기 1
    if (v.y > 0) {
      // 송장이 붙는 가운데는 주름을 눌러 둔다
      const calm = 0.3 + 0.7 * Math.min(1, (nx * nx + nz * nz) * 2.5);
      y += 0.07 * puff * calm * (Math.sin(nx * 6.3 + p1) * Math.sin(nz * 4.1 + p2) + 0.5 * Math.sin((nx + nz) * 9 + p3));
    }
    const shrink = 1 - 0.05 * (1 - puff); // 가장자리는 안으로 접혀 들어간다
    pos.setXYZ(i, v.x * w * shrink, y * h, v.z * d * shrink);
  }
  g.computeVertexNormals();
  return g;
}

// ---------------------------------------------------------------------------
// 장면
// ---------------------------------------------------------------------------
function buildWorld(renderer) {
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const rand = seeded(20261116);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(BG, 0.036);
  scene.environment = studioEnvironment(renderer);
  scene.environmentIntensity = 0.6;

  scene.add(new THREE.HemisphereLight(0x8aa0c0, 0x08090b, 0.5));
  const key = new THREE.DirectionalLight(0xfff3e8, 1.5);
  key.position.set(-6, 11, 7);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -14, right: 14, top: 10, bottom: -10, near: 1, far: 40 });
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 3;
  scene.add(key);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(120, 120),
    new THREE.MeshStandardMaterial({ map: floorTexture(aniso), roughness: 0.62, metalness: 0 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // 천장 조명 줄과 먼 기둥 — 안개 속으로 사라지며 공간의 깊이를 잡는다
  const ceilingMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xffffff).multiplyScalar(2.2) });
  const fixtures = [];
  for (const z of [-7.2, -3.6, 0, 3.6]) for (let x = -20; x <= 20; x += 5) fixtures.push([x, 7.2, z]);
  scene.add(instanced(new THREE.BoxGeometry(3, 0.05, 0.16), ceilingMat, fixtures));
  const pillars = [];
  for (let x = -21; x <= 21; x += 7) pillars.push([x, 4, -11.5]);
  scene.add(instanced(new THREE.BoxGeometry(0.5, 8, 0.5), new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.8 }), pillars));

  const aluminum = new THREE.MeshStandardMaterial({ color: 0x9ea4ad, metalness: 0.9, roughness: 0.32 });
  const steel = new THREE.MeshStandardMaterial({ color: 0xc9ced6, metalness: 1, roughness: 0.22 });
  const darkBed = new THREE.MeshStandardMaterial({ color: 0x0e0f12, roughness: 0.9 });
  const glow = new THREE.MeshBasicMaterial({ color: ORANGE, toneMapped: false });
  const bagColors = [0xdf6a2c, 0xe27532, 0xd45f27, 0xe8823f].map(
    (color) => new THREE.MeshPhysicalMaterial({ color, roughness: 0.42, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.3 }),
  );
  const labelMats = Array.from({ length: 8 }, () => {
    const map = labelTexture(rand, aniso);
    // 어두운 곳에서도 송장이 읽히게 살짝 스스로 빛낸다
    return new THREE.MeshStandardMaterial({ map, roughness: 0.55, emissive: 0xffffff, emissiveMap: map, emissiveIntensity: 0.16 });
  });
  const bagShapes = Array.from({ length: 12 }, () => {
    const w = 0.42 + rand() * 0.22;
    const d = 0.32 + rand() * 0.16;
    const h = 0.07 + rand() * 0.07;
    return { geo: bagGeometry(w, h, d, rand), w, d, h };
  });

  const legs = [];
  const feet = [];
  const braces = [];
  const posts = [];
  const belts = [];
  const bags = [];
  const dash = dashTexture(aniso);

  function belt(z, speed, main) {
    const tex = rubberTexture(aniso);
    const rubber = new THREE.MeshStandardMaterial({ map: tex, bumpMap: tex, bumpScale: 0.4, roughness: 0.82 });
    const surface = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.03, 1.5), [darkBed, darkBed, rubber, darkBed, darkBed, darkBed]);
    surface.position.set(0, BELT_TOP - 0.015, z);
    surface.receiveShadow = true;
    scene.add(surface);
    const bed = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.1, 1.52), darkBed);
    bed.position.set(0, BELT_TOP - 0.09, z);
    scene.add(bed);
    for (const side of [-1, 1]) {
      // 알루미늄 옆판 — 반사 환경이 길게 맺힌다
      const frame = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.2, 0.05), aluminum);
      frame.position.set(0, BELT_TOP - 0.06, z + side * 0.79);
      frame.castShadow = frame.receiveShadow = true;
      scene.add(frame);
      // 둥근 가이드 레일
      const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, BELT_LEN, 12), steel);
      rail.rotation.z = Math.PI / 2;
      rail.position.set(0, BELT_TOP + 0.1, z + side * 0.74);
      rail.castShadow = true;
      scene.add(rail);
      for (let x = -BELT_LEN / 2 + 0.75; x < BELT_LEN / 2; x += 1.5) posts.push([x, BELT_TOP + 0.05, z + side * 0.74]);
      if (main) {
        // 우리 라인만 옆판 바깥에 주황 띠 — 멀리서도 어느 라인 이야기인지 보인다
        const strip = new THREE.Mesh(new THREE.BoxGeometry(BELT_LEN, 0.014, 0.012), glow);
        strip.position.set(0, BELT_TOP - 0.02, z + side * 0.822);
        scene.add(strip);
      }
      // 바닥 차선
      const lane = new THREE.Mesh(
        new THREE.PlaneGeometry(BELT_LEN, main ? 0.06 : 0.035),
        new THREE.MeshBasicMaterial({ map: dash, color: main ? ORANGE : 0x5a5f68, transparent: true, opacity: main ? 0.75 : 0.35, depthWrite: false }),
      );
      lane.rotation.x = -Math.PI / 2;
      lane.position.set(0, 0.003, z + side * 1.35);
      scene.add(lane);
    }
    for (let x = -BELT_LEN / 2 + 1; x < BELT_LEN / 2; x += 3) {
      for (const side of [-1, 1]) {
        legs.push([x, (BELT_TOP - 0.16) / 2, z + side * 0.68]);
        feet.push([x, 0.01, z + side * 0.68]);
      }
      braces.push([x, 0.24, z]);
    }
    belts.push({ tex, speed });

    let x = -BELT_LEN / 2 + rand() * 2;
    while (x < BELT_LEN / 2) {
      const shape = bagShapes[Math.floor(rand() * bagShapes.length)];
      const bag = new THREE.Group();
      const body = new THREE.Mesh(shape.geo, bagColors[Math.floor(rand() * bagColors.length)]);
      body.castShadow = body.receiveShadow = true;
      bag.add(body);
      const lw = Math.min(shape.w, shape.d) * 0.72;
      const label = new THREE.Mesh(new THREE.PlaneGeometry(lw, lw / 1.5), labelMats[Math.floor(rand() * labelMats.length)]);
      label.rotation.x = -Math.PI / 2;
      // 일부 송장은 비뚤게 붙어 있다 — 우리가 풀려는 문제가 이것이다
      label.rotation.z = rand() < 0.35 ? (rand() - 0.5) * 0.9 : (rand() - 0.5) * 0.12;
      label.position.y = shape.h * 1.04;
      label.receiveShadow = true;
      bag.add(label);
      bag.rotation.y = (rand() - 0.5) * 0.7;
      bag.rotation.x = (rand() - 0.5) * 0.06;
      bag.position.set(x, BELT_TOP, z + (rand() - 0.5) * 0.4);
      scene.add(bag);
      bags.push({ obj: bag, speed });
      x += shape.w + 0.55 + rand() * 1.1;
    }
  }

  belt(0, 1.2, true);
  belt(-3.6, 1.0, false);
  belt(3.6, 1.4, false);
  belt(-7.2, 1.1, false);

  const legGeo = new THREE.BoxGeometry(0.06, BELT_TOP - 0.16, 0.06);
  scene.add(instanced(legGeo, aluminum, legs, true));
  scene.add(instanced(new THREE.BoxGeometry(0.14, 0.02, 0.14), darkBed, feet));
  scene.add(instanced(new THREE.BoxGeometry(0.04, 0.04, 1.36), aluminum, braces, true));
  scene.add(instanced(new THREE.CylinderGeometry(0.008, 0.008, 0.1, 8), steel, posts));

  // 옆 라인의 기존 스캐너 — 흰 표시등, 우리 것과 구분
  const anodized = new THREE.MeshStandardMaterial({ color: 0x23262c, metalness: 0.75, roughness: 0.35 });
  for (const z of [-3.6, 3.6]) {
    const arch = new THREE.Group();
    for (const side of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.1, 2.0, 0.1), anodized);
      post.position.set(0, 1.0, side * 1.05);
      arch.add(post);
    }
    const beam = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.14, 2.2), anodized);
    beam.position.y = 2.0;
    arch.add(beam);
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.012, 1.6), new THREE.MeshBasicMaterial({ color: 0xdfe7ff }));
    lamp.position.y = 1.925;
    arch.add(lamp);
    arch.position.set(z > 0 ? 5 : -5, 0, z);
    arch.traverse((o) => (o.castShadow = true));
    scene.add(arch);
  }

  // 우리 판독 카메라 — x=0, 가운데 라인 위
  const gantry = new THREE.Group();
  for (const side of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.12, GANTRY_H, 0.12), anodized);
    post.position.set(0, GANTRY_H / 2, side * 1.15);
    gantry.add(post);
  }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.16, 2.5), anodized);
  beam.position.y = GANTRY_H;
  gantry.add(beam);
  const beamGlow = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.01, 2.2), glow);
  beamGlow.position.y = GANTRY_H - 0.085;
  gantry.add(beamGlow);
  gantry.traverse((o) => (o.castShadow = true));
  scene.add(gantry);

  const rig = new THREE.Group();
  rig.position.set(0, RIG_Y, 0);
  const housing = new THREE.Mesh(
    new THREE.BoxGeometry(0.42, 0.26, 0.3),
    new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.3, metalness: 0.5 }),
  );
  housing.castShadow = true;
  rig.add(housing);
  const mountLen = GANTRY_H - 0.08 - (RIG_Y + 0.13);
  const mount = new THREE.Mesh(new THREE.BoxGeometry(0.06, mountLen, 0.06), anodized);
  mount.position.y = 0.13 + mountLen / 2;
  rig.add(mount);
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.085, 0.16, 48), aluminum);
  barrel.position.y = -0.21;
  rig.add(barrel);
  const glass = new THREE.Mesh(
    new THREE.CircleGeometry(0.068, 48),
    new THREE.MeshPhysicalMaterial({ color: 0x0a1626, roughness: 0.04, metalness: 0.1, clearcoat: 1, envMapIntensity: 2.2 }),
  );
  glass.rotation.x = Math.PI / 2;
  glass.position.y = -0.291;
  rig.add(glass);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.083, 0.012, 12, 48), glow);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -0.285;
  rig.add(ring);
  const ledMat = new THREE.MeshBasicMaterial({ color: ORANGE, toneMapped: false });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), ledMat);
  led.position.set(0.15, 0.06, 0.152);
  rig.add(led);
  // 카메라 몸통의 PIMM. 명판 — 다가가면서 읽힌다
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(0.24, 0.06),
    new THREE.MeshBasicMaterial({
      map: canvasTexture(512, 128, (g) => {
        g.fillStyle = "#15171b";
        g.fillRect(0, 0, 512, 128);
        g.font = "800 76px Pretendard, system-ui, sans-serif";
        g.textBaseline = "middle";
        const parts = [["P", "#f4f2ee"], ["I", "#df6a2c"], ["MM", "#f4f2ee"], [".", "#df6a2c"]];
        const total = parts.reduce((s, [t]) => s + g.measureText(t).width, 0);
        let px = (512 - total) / 2;
        for (const [t, c] of parts) {
          g.fillStyle = c;
          g.fillText(t, px, 66);
          px += g.measureText(t).width;
        }
      }, aniso),
    }),
  );
  plate.position.set(-0.03, 0.0, 0.151);
  rig.add(plate);
  scene.add(rig);

  // 판독 카메라 아래를 비추는 조명 — 지나가는 택배가 무대 위에 올라선 듯 밝아진다
  const spot = new THREE.SpotLight(0xffe6d4, 22, 6, 0.6, 0.75, 2);
  spot.position.set(0, LENS_Y, 0);
  spot.target.position.set(0, 0, 0);
  scene.add(spot, spot.target);
  const warm = new THREE.PointLight(ORANGE, 4, 3.2, 2);
  warm.position.set(0, LENS_Y - 0.1, 0);
  scene.add(warm);

  // 판독 영역 — 렌즈에서 벨트까지 퍼지는 주황 빛기둥. 마지막 시점(시야각 60°)의 넓이와 맞춘다
  const coneH = LENS_Y - BELT_TOP;
  const halfZ = coneH * Math.tan((30 * Math.PI) / 180);
  const halfX = halfZ * 1.6;
  const coneGeo = new THREE.ConeGeometry(halfX * Math.SQRT2, coneH, 4, 1, true);
  coneGeo.rotateY(Math.PI / 4);
  const coneMat = new THREE.MeshBasicMaterial({
    color: ORANGE, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false,
    blending: THREE.AdditiveBlending, toneMapped: false,
  });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  cone.scale.set(1, 1, halfZ / halfX);
  cone.position.set(0, BELT_TOP + coneH / 2, 0);
  scene.add(cone);

  // 빛기둥 속을 떠도는 먼지 — 빛이 공기 중에 있다는 느낌
  const dustN = 280;
  const dustPos = new Float32Array(dustN * 3);
  for (let i = 0; i < dustN; i++) {
    dustPos[i * 3] = (rand() - 0.5) * 2.6;
    dustPos[i * 3 + 1] = BELT_TOP + 0.15 + rand() * (coneH - 0.15);
    dustPos[i * 3 + 2] = (rand() - 0.5) * 1.6;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(
    dustGeo,
    new THREE.PointsMaterial({ color: 0xffc9a8, size: 0.012, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  scene.add(dust);

  // 벨트 위를 오가는 판독선
  const laser = new THREE.Group();
  const lineMat = new THREE.MeshBasicMaterial({ color: ORANGE, toneMapped: false, blending: THREE.AdditiveBlending, transparent: true });
  const haloMat = new THREE.MeshBasicMaterial({
    color: ORANGE, transparent: true, opacity: 0.28, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
  });
  for (const [w, mat] of [[0.016, lineMat], [0.18, haloMat]]) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, 1.32), mat);
    m.rotation.x = -Math.PI / 2;
    laser.add(m);
  }
  laser.position.y = BELT_TOP + 0.012;
  scene.add(laser);

  let clock = 0;
  return {
    scene,
    rigPosition: rig.position,
    update(dt) {
      clock += dt;
      for (const b of belts) b.tex.offset.x -= (b.speed * dt) / 0.6;
      for (const b of bags) {
        b.obj.position.x += b.speed * dt;
        if (b.obj.position.x > BELT_LEN / 2) b.obj.position.x -= BELT_LEN;
      }
      laser.position.x = Math.sin(clock * 2.4) * 0.62;
      dust.rotation.y = clock * 0.05;
      dust.position.y = Math.sin(clock * 0.7) * 0.03;
      ledMat.color.setHex(Math.sin(clock * 7) > 0 ? ORANGE : 0x5a2810);
    },
    // 렌즈 속으로 들어갈 때 카메라 몸통과 빛기둥을 걷는다 — 안쪽 면이 화면을 가린다
    approach(u, distToRig) {
      coneMat.opacity = 0.12 * (1 - smooth(0.86, 0.97, u));
      rig.visible = distToRig > 0.34;
    },
  };
}

function instanced(geo, mat, positions, shadow = false) {
  const mesh = new THREE.InstancedMesh(geo, mat, positions.length);
  const m = new THREE.Matrix4();
  positions.forEach((p, i) => mesh.setMatrixAt(i, m.makeTranslation(p[0], p[1], p[2])));
  mesh.castShadow = shadow;
  mesh.receiveShadow = shadow;
  return mesh;
}

// 카메라가 지나는 길. 구심(centripetal) 곡선이라 꺾이는 곳에서도 튀지 않는다.
// 마지막 다가감은 정면(+z, 명판이 붙은 쪽)에서 한다 — 비스듬히 다가가면 아래를 내려다보며
// 렌즈로 들어가는 순간 화면이 45° 가까이 돌아간다.
const PATH = new THREE.CatmullRomCurve3(
  [
    new THREE.Vector3(-10.5, 5.4, 8.6),
    new THREE.Vector3(-6.0, 4.4, 5.6),
    new THREE.Vector3(-2.2, 3.7, 3.0),
    new THREE.Vector3(-0.08, RIG_Y + 0.14, 0.95),
    new THREE.Vector3(-0.05, RIG_Y - 0.22, 0.28),
    new THREE.Vector3(0, LENS_Y - 0.04, 0.02),
  ],
  false,
  "centripetal",
);
const GAZE = new THREE.CatmullRomCurve3(
  [
    new THREE.Vector3(1.6, 0.9, -1.4),
    new THREE.Vector3(0.6, 1.45, -0.5),
    new THREE.Vector3(0.02, RIG_Y - 0.55, 0),
    new THREE.Vector3(0, RIG_Y - 0.3, 0),
    new THREE.Vector3(0, 1.45, -0.03),
    new THREE.Vector3(0, BELT_TOP, 0),
  ],
  false,
  "centripetal",
);

const FONT = 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif';
const CSS = `
.pimm-intro { position: fixed; z-index: 1000; left: 0; top: 0; width: 100vw; height: 100vh; overflow: hidden;
  background: #0c0e11; cursor: pointer; will-change: left, top, width, height, opacity; }
.pimm-intro canvas { position: absolute; display: block; }
.pi-vignette { position: absolute; inset: 0; pointer-events: none; transition: opacity .4s ease;
  background: radial-gradient(120% 95% at 50% 45%, transparent 55%, rgba(0,0,0,.55) 100%); }
.pi-logo { position: absolute; left: clamp(24px, 4vw, 56px); bottom: clamp(22px, 5vh, 48px); pointer-events: none;
  font: 800 26px/1 ${FONT}; letter-spacing: -.045em; color: #f4f2ee;
  opacity: 0; transform: translateY(8px); transition: opacity .6s ease, transform .8s cubic-bezier(.2,.8,.2,1); }
.pi-logo em, .pi-logo b { font-style: normal; color: #df6a2c; }
.pimm-intro.titled .pi-logo { opacity: 1; transform: none; }
.pi-finder { position: absolute; inset: 10% 8%; opacity: 0; transition: opacity .5s ease; pointer-events: none; }
.pimm-intro.pov .pi-finder { opacity: 1; }
.pi-finder i { position: absolute; width: 34px; height: 34px; border: 2.5px solid #df6a2c; }
.pi-finder i:nth-child(1) { left: 0; top: 0; border-right: 0; border-bottom: 0; }
.pi-finder i:nth-child(2) { right: 0; top: 0; border-left: 0; border-bottom: 0; }
.pi-finder i:nth-child(3) { left: 0; bottom: 0; border-right: 0; border-top: 0; }
.pi-finder i:nth-child(4) { right: 0; bottom: 0; border-left: 0; border-top: 0; }
.pi-cam { position: absolute; left: 4px; top: -30px; display: flex; align-items: center; gap: 8px;
  font: 600 12px/1 ${FONT}; letter-spacing: .1em; color: #fff; text-shadow: 0 1px 3px rgba(0,0,0,.55); }
.pi-cam::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #df6a2c; box-shadow: 0 0 0 4px rgba(223,106,44,.32); }
.pi-scan { position: absolute; left: 0; right: 0; top: 0; height: 2px; opacity: 0; pointer-events: none;
  background: linear-gradient(90deg, transparent, #df6a2c 18%, #ffb07f 50%, #df6a2c 82%, transparent);
  box-shadow: 0 0 22px 5px rgba(223,106,44,.45); }
.pimm-intro.flash .pi-scan { animation: pi-scan .6s cubic-bezier(.4,0,.2,1) forwards; }
@keyframes pi-scan { 0% { top: 0; opacity: 0; } 12% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
.pi-skip { position: absolute; right: clamp(24px, 4vw, 56px); bottom: clamp(22px, 5vh, 48px);
  appearance: none; background: none; border: 0; padding: 6px 2px; cursor: pointer;
  font: 600 13px/1 ${FONT}; letter-spacing: .16em; color: rgba(255,255,255,.6); transition: color .2s, opacity .3s; }
.pi-skip:hover, .pi-skip:focus-visible { color: #fff; outline: none; }
.pi-skip::after { content: ""; display: block; height: 1px; margin-top: 5px; background: #df6a2c;
  transform: scaleX(0); transform-origin: left; transition: transform .25s ease; }
.pi-skip:hover::after, .pi-skip:focus-visible::after { transform: scaleX(1); }
.pimm-intro.landing .pi-skip, .pimm-intro.landing .pi-logo, .pimm-intro.landing .pi-vignette { opacity: 0; }
`;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * @param {{ targetRect: () => ({left:number, top:number, width:number, height:number}) | null, onLand: () => void }} opts
 *   targetRect — 장면이 줄어들어 내려앉을 자리. 매 프레임 다시 묻는다(그 사이 화면이 움직여도 맞는다)
 *   onLand — 작업 화면을 띄울 때 부른다. 실패해도 반드시 한 번은 불린다.
 */
export async function runIntro({ targetRect, onLand }) {
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
    return false;
  }
  // 송장 숫자와 명판을 Pretendard 로 그리려면 글꼴이 먼저 와 있어야 한다. 늦으면 기다리지 않는다
  await Promise.race([document.fonts?.load("800 76px Pretendard").catch(() => {}), wait(400)]);

  return new Promise((resolve) => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.className = "pimm-intro";
    root.innerHTML =
      '<div class="pi-vignette" aria-hidden="true"></div>' +
      '<div class="pi-finder" aria-hidden="true"><i></i><i></i><i></i><i></i><span class="pi-cam">CAM-01 · SCAN</span></div>' +
      '<div class="pi-scan" aria-hidden="true"></div>' +
      '<div class="pi-logo" aria-hidden="true">P<em>I</em>MM<b>.</b></div>' +
      '<button type="button" class="pi-skip">skip</button>';
    root.prepend(renderer.domElement);
    document.body.appendChild(root);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setClearColor(BG, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
    let bufW = 0;
    let bufH = 0;
    let morphing = false;
    let skipping = false;
    let finished = false;
    let raf = 0;
    let t = 0;
    let last = performance.now();
    let flashed = false;
    // 카메라는 길 위의 목표점을 살짝 늦게 따라간다 — 출발·정지와 꺾임이 부드러워진다
    const eye = PATH.getPoint(0);
    const gaze = GAZE.getPoint(0);
    const goalEye = new THREE.Vector3();
    const goalGaze = new THREE.Vector3();

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
    // 그리는 크기는 화면 전체로 두고, 줄어드는 동안은 자리를 꽉 채우도록(cover) 놓는다 — 찌그러지지 않는다
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
      const r = targetRect && targetRect();
      if (!r || r.width < 4 || r.height < 4) return;
      const w = lerp(window.innerWidth, r.width, k);
      const h = lerp(window.innerHeight, r.height, k);
      const s = root.style;
      s.left = lerp(0, r.left, k) + "px";
      s.top = lerp(0, r.top, k) + "px";
      s.width = w + "px";
      s.height = h + "px";
      s.borderRadius = 16 * k + "px";
      s.boxShadow = `0 ${18 * k}px ${44 * k}px -${16 * k}px rgba(0,0,0,${0.6 * k})`;
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

      const u = easeInOutSine(clamp01((t - START) / TRAVEL));
      const k = PATH.getUtoTmapping(u);
      goalEye.copy(PATH.getPoint(k));
      goalGaze.copy(GAZE.getPoint(k));
      eye.lerp(goalEye, 1 - Math.exp(-dt * 7));
      gaze.lerp(goalGaze, 1 - Math.exp(-dt * 6));
      camera.position.copy(eye);
      camera.lookAt(gaze);
      camera.fov = 42 + 18 * smooth(0.8, 1, u);
      camera.updateProjectionMatrix();
      world.approach(u, eye.distanceTo(world.rigPosition));

      root.classList.toggle("titled", t > 0.25 && u < 0.86);
      if (u > 0.96) root.classList.add("pov"); // 렌즈 속에 들어간 뒤에야 조준 괄호를 띄운다
      if (!flashed && t > START + TRAVEL - 0.12) {
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
        world?.scene.environment?.dispose();
        world?.scene.traverse((o) => {
          o.geometry?.dispose();
          for (const m of [].concat(o.material || [])) {
            for (const key of ["map", "bumpMap", "emissiveMap"]) m[key]?.dispose();
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

    function start() {
      if (finished) return;
      raf = requestAnimationFrame((now) => {
        last = now;
        frame(now);
      });
    }

    try {
      resize();
      window.addEventListener("resize", resize);
      window.addEventListener("keydown", onKey);
      root.addEventListener("click", skip);
      root.querySelector(".pi-skip").focus({ preventScroll: true });
      camera.position.copy(eye);
      camera.lookAt(gaze);
      // 셰이더를 먼저 굽는다 — 첫 프레임에 굽느라 멈칫하면 시작이 끊겨 보인다.
      // 굽는 동안은 어두운 가림막이 보이고, 다 구운 뒤에 시계가 간다.
      renderer.compileAsync(world.scene, camera).then(start, start);
    } catch {
      finish();
    }
  });
}
