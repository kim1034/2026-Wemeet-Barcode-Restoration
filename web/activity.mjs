// A cheap upload gate, never a barcode detector. The reference stays fixed so a
// hand-held object does not disappear into the background when movement stops.
export class ActivityDetector {
  constructor({ pixelDelta = 18, changedRatio = 0.025, releaseMs = 300 } = {}) {
    Object.assign(this, { pixelDelta, changedRatio, releaseMs });
    this.reset();
  }
  reset() {
    this.background = null;
    this.previous = null;
    this.streak = 0;
    this.lastActive = -Infinity;
  }
  update(gray, now) {
    if (!this.background || this.background.length !== gray.length) {
      this.background = Uint8Array.from(gray);
      this.previous = Uint8Array.from(gray);
      return { active: false, changedRatio: 0, calibrated: true };
    }
    let changed = 0, motion = 0;
    for (let i = 0; i < gray.length; i++) {
      if (Math.abs(gray[i] - this.background[i]) > this.pixelDelta) changed++;
      if (Math.abs(gray[i] - this.previous[i]) > this.pixelDelta) motion++;
    }
    const ratio = Math.max(changed, motion) / gray.length;
    this.previous.set(gray);
    this.streak = ratio >= this.changedRatio ? this.streak + 1 : 0;
    if (this.streak >= 2) this.lastActive = now;
    return { active: now - this.lastActive < this.releaseMs, changedRatio: ratio, calibrated: false };
  }
}
