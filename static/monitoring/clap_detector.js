/**
 * MedinaMind — Double-clap detector (Web Audio API)
 * Detects two sharp transients within ~0.5–1.2s.
 */
(function (global) {
  "use strict";

  function ClapDetector(options) {
    this.onDoubleClap = options.onDoubleClap || function () {};
    this.onClap = options.onClap || function () {};
    this.onError = options.onError || function () {};
    this.threshold = options.threshold != null ? options.threshold : 0.42;
    this.minClapGap = options.minClapGap || 180;
    this.maxClapGap = options.maxClapGap || 900;
    this.pairWindow = options.pairWindow || 1200;
    this.cooldownMs = options.cooldownMs || 3500;
    this.minAbsolutePeak = options.minAbsolutePeak != null ? options.minAbsolutePeak : 0.34;
    this.minSharpness = options.minSharpness != null ? options.minSharpness : 2.6;
    this.noiseFloor = 0.02;

    this._ctx = null;
    this._analyser = null;
    this._stream = null;
    this._timeData = null;
    this._freqData = null;
    this._raf = 0;
    this._running = false;
    this._enabled = false;
    this._lastPeak = 0;
    this._firstClapAt = 0;
    this._clapCount = 0;
    this._cooldownUntil = 0;
    this._calibrating = true;
    this._calibrationStart = 0;
    this._calibrationSamples = [];
    this._recentPeaks = [];
  }

  ClapDetector.prototype._analyze = function () {
    if (!this._analyser || !this._timeData) {
      return { level: 0, peak: 0, rms: 0, hf: 0, sharpness: 0 };
    }

    this._analyser.getByteTimeDomainData(this._timeData);
    var peak = 0;
    var sumSq = 0;
    for (var i = 0; i < this._timeData.length; i++) {
      var v = Math.abs(this._timeData[i] - 128) / 128;
      if (v > peak) peak = v;
      sumSq += v * v;
    }
    var rms = Math.sqrt(sumSq / this._timeData.length);

    var hf = 0;
    if (this._freqData) {
      this._analyser.getByteFrequencyData(this._freqData);
      var start = 8;
      var end = Math.min(64, this._freqData.length);
      var hfSum = 0;
      for (var j = start; j < end; j++) hfSum += this._freqData[j];
      hf = hfSum / ((end - start) * 255);
    }

    var level = Math.max(peak, rms * 1.8, hf * 0.95);
    var sharpness = peak / Math.max(rms, 0.008);
    return { level: level, peak: peak, rms: rms, hf: hf, sharpness: sharpness };
  };

  ClapDetector.prototype._peakLevel = function () {
    return this._analyze().level;
  };

  ClapDetector.prototype._isClapLike = function (sample) {
    if (!sample) return false;
    if (sample.peak < this.minAbsolutePeak) return false;
    if (sample.sharpness < this.minSharpness) return false;
    if (sample.hf < this.noiseFloor * 0.85 && sample.peak < this.minAbsolutePeak + 0.08) return false;
    return true;
  };

  ClapDetector.prototype._registerPeak = function (now, level) {
    if (this._clapCount === 0) {
      this._firstClapAt = now;
      this._clapCount = 1;
      try {
        this.onClap(1);
      } catch (e0) {}
      return;
    }

    var gap = now - this._lastPeak;
    if (gap >= this.minClapGap && now - this._firstClapAt <= this.pairWindow) {
      this._clapCount += 1;
      try {
        this.onClap(2);
      } catch (e1) {}
      if (this._clapCount >= 2) {
        this._cooldownUntil = now + this.cooldownMs;
        this._clapCount = 0;
        this._firstClapAt = 0;
        try {
          this.onDoubleClap();
        } catch (err) {
          this.onError(err);
        }
      }
    } else if (now - this._firstClapAt > this.pairWindow) {
      this._firstClapAt = now;
      this._clapCount = 1;
      try {
        this.onClap(1);
      } catch (e2) {}
    }

    this._lastPeak = now;
    this._recentPeaks.push(now);
    if (this._recentPeaks.length > 6) this._recentPeaks.shift();
  };

  ClapDetector.prototype._tick = function () {
    if (!this._running) return;
    var now = performance.now();
    var level = this._peakLevel();

    if (this._calibrating) {
      this._calibrationSamples.push(level);
      if (now - this._calibrationStart > 2000) {
        this._calibrating = false;
        var sorted = this._calibrationSamples.slice().sort(function (a, b) {
          return a - b;
        });
        var median = sorted[Math.floor(sorted.length / 2)] || 0.04;
        this.noiseFloor = Math.min(0.1, Math.max(0.02, median * 1.35));
        this.threshold = Math.max(this.threshold, this.noiseFloor + 0.16);
      }
    } else if (now >= this._cooldownUntil) {
      var sample = this._analyze();
      var level = sample.level;
      var dynamicThreshold = Math.max(this.threshold, this.noiseFloor + 0.14);
      var rising = level >= dynamicThreshold;
      var cooled = now - this._lastPeak > 90;
      var clapLike = this._isClapLike(sample);

      if (rising && cooled && clapLike) {
        this._registerPeak(now, level);
        if (this._clapCount <= 1) this._lastPeak = now;
      } else if (this._clapCount > 0 && now - this._firstClapAt > this.pairWindow) {
        this._clapCount = 0;
        this._firstClapAt = 0;
      }
    }

    this._raf = requestAnimationFrame(this._tick.bind(this));
  };

  ClapDetector.prototype.start = function () {
    var self = this;
    if (this._enabled && this._running) return Promise.resolve();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return Promise.reject(new Error("Microphone not supported in this browser."));
    }

    if (this._stream) {
      this._running = true;
      this._enabled = true;
      this._calibrating = true;
      this._calibrationStart = performance.now();
      this._calibrationSamples = [];
      this._tick();
      if (this._ctx && this._ctx.state === "suspended") {
        return this._ctx.resume();
      }
      return Promise.resolve();
    }

    return navigator.mediaDevices
      .getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      })
      .then(function (stream) {
        self._stream = stream;
        var AC = window.AudioContext || window.webkitAudioContext;
        self._ctx = new AC();
        var src = self._ctx.createMediaStreamSource(stream);
        self._analyser = self._ctx.createAnalyser();
        self._analyser.fftSize = 2048;
        self._analyser.smoothingTimeConstant = 0.05;
        src.connect(self._analyser);
        self._timeData = new Uint8Array(self._analyser.fftSize);
        self._freqData = new Uint8Array(self._analyser.frequencyBinCount);
        self._running = true;
        self._enabled = true;
        self._calibrating = true;
        self._calibrationStart = performance.now();
        self._calibrationSamples = [];
        self._clapCount = 0;
        self._firstClapAt = 0;
        if (self._ctx.state === "suspended") {
          return self._ctx.resume();
        }
      })
      .then(function () {
        self._tick();
      })
      .catch(function (err) {
        self.onError(err);
        throw err;
      });
  };

  ClapDetector.prototype.pause = function () {
    this._running = false;
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = 0;
    }
  };

  ClapDetector.prototype.resume = function () {
    if (!this._enabled || this._running) return Promise.resolve();
    this._running = true;
    this._calibrating = true;
    this._calibrationStart = performance.now();
    this._calibrationSamples = [];
    this._clapCount = 0;
    this._firstClapAt = 0;
    var p = Promise.resolve();
    if (this._ctx && this._ctx.state === "suspended") {
      p = this._ctx.resume();
    }
    var self = this;
    return p.then(function () {
      self._tick();
    });
  };

  ClapDetector.prototype.stop = function () {
    this._running = false;
    this._enabled = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._stream) {
      this._stream.getTracks().forEach(function (t) {
        t.stop();
      });
    }
    if (this._ctx) {
      try {
        this._ctx.close();
      } catch (e0) {}
    }
    this._stream = null;
    this._ctx = null;
    this._analyser = null;
    this._raf = 0;
  };

  ClapDetector.prototype.isEnabled = function () {
    return this._enabled;
  };

  global.MedinaMindClapDetector = ClapDetector;
})(window);
