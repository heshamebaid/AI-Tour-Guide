/**
 * Talk to Pharos - Voice-to-Voice module
 * Handles: speech recognition (user), speech synthesis (pharaoh), and the voice loop.
 * Load before talk_to_pharos.js
 */
(function (global) {
  "use strict";

  const SpeechRecognition = global.SpeechRecognition || global.webkitSpeechRecognition;
  const canUseSpeechRecognition = Boolean(SpeechRecognition);
  const canUseSpeechSynthesis = "speechSynthesis" in global;

  let recognition = null;
  let isListening = false;
  let pharaohVoice = null;
  let autoSendTimeout = null;

  /** Callbacks set by the main app */
  let callbacks = {
    onFinalTranscript: null,
    onSpeakEnd: null,
    onError: null,
    onListeningChange: null,
  };

  function findBestPharaohVoice() {
    if (!canUseSpeechSynthesis) return null;
    const voices = global.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;

    const preferred = ["male", "deep", "low", "david", "daniel", "james", "rich"];
    for (const voice of voices) {
      const name = voice.name.toLowerCase();
      if (voice.lang.startsWith("en") && preferred.some((k) => name.includes(k))) return voice;
    }

    const en = voices.filter((v) => v.lang.startsWith("en"));
    const female = ["susan", "karen", "zira", "samantha", "victoria", "kate", "sarah"];
    const male = en.filter((v) => !female.some((k) => v.name.toLowerCase().includes(k)));
    return (male.length ? male : en)[0] || voices[0];
  }

  function initPharaohVoice() {
    if (!canUseSpeechSynthesis) return;
    if (global.speechSynthesis.getVoices().length === 0) {
      global.speechSynthesis.addEventListener("voiceschanged", () => {
        pharaohVoice = findBestPharaohVoice();
      }, { once: true });
    } else {
      pharaohVoice = findBestPharaohVoice();
    }
  }

  /**
   * Speak pharaoh response. When finished, calls callbacks.onSpeakEnd (for voice-to-voice loop).
   */
  function speak(text, options) {
    if (!canUseSpeechSynthesis || !text) {
      if (options && options.onEnd) options.onEnd();
      return;
    }
    try {
      global.speechSynthesis.cancel();
      if (!pharaohVoice) pharaohVoice = findBestPharaohVoice();

      const utterance = new global.SpeechSynthesisUtterance(text);
      utterance.rate = options && options.rate != null ? options.rate : 0.85;
      utterance.pitch = options && options.pitch != null ? options.pitch : 0.7;
      utterance.volume = 0.95;
      utterance.lang = "en-US";
      if (pharaohVoice) utterance.voice = pharaohVoice;

      utterance.onstart = () => {
        if (options && options.onStart) options.onStart();
      };
      utterance.onend = () => {
        if (callbacks.onSpeakEnd) callbacks.onSpeakEnd();
        if (options && options.onEnd) options.onEnd();
      };
      utterance.onerror = () => {
        if (callbacks.onSpeakEnd) callbacks.onSpeakEnd();
        if (options && options.onEnd) options.onEnd();
      };

      global.speechSynthesis.speak(utterance);
    } catch (err) {
      if (callbacks.onError) callbacks.onError("speech", err);
      if (options && options.onEnd) options.onEnd();
    }
  }

  function stopListening() {
    if (recognition && isListening) recognition.stop();
    isListening = false;
    if (autoSendTimeout) {
      clearTimeout(autoSendTimeout);
      autoSendTimeout = null;
    }
    if (global.speechSynthesis.speaking) global.speechSynthesis.cancel();
    if (callbacks.onListeningChange) callbacks.onListeningChange(false);
  }

  function startListening(pauseMsBeforeSend) {
    if (!recognition) return false;
    if (isListening) {
      stopListening();
      return false;
    }
    const pause = pauseMsBeforeSend == null ? 800 : pauseMsBeforeSend;
    isListening = true;
    if (callbacks.onListeningChange) callbacks.onListeningChange(true);

    recognition.onresult = function (event) {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += t + " ";
        else interimText += t;
      }
      const combined = (finalText + interimText).trim();
      if (callbacks.onInterimTranscript) callbacks.onInterimTranscript(combined);

      if (finalText.trim()) {
        if (autoSendTimeout) clearTimeout(autoSendTimeout);
        autoSendTimeout = setTimeout(() => {
          autoSendTimeout = null;
          const toSend = (finalText + interimText).trim();
          if (toSend && callbacks.onFinalTranscript) callbacks.onFinalTranscript(toSend);
        }, pause);
      }
    };

    recognition.onend = function () {
      if (isListening && recognition.continuous && !global.speechSynthesis.speaking) {
        try { recognition.start(); } catch (e) { /* ignore */ }
      }
    };

    recognition.onerror = function (event) {
      const msg = event.error === "no-speech" ? "No speech detected."
        : event.error === "not-allowed" ? "Microphone permission denied."
        : event.error === "audio-capture" ? "Microphone not found."
        : "Voice error: " + event.error;
      if (callbacks.onError) callbacks.onError("recognition", msg);
      stopListening();
    };

    try {
      recognition.start();
      return true;
    } catch (e) {
      isListening = false;
      if (callbacks.onListeningChange) callbacks.onListeningChange(false);
      if (callbacks.onError) callbacks.onError("recognition", e.message);
      return false;
    }
  }

  function init(config) {
    callbacks = Object.assign({}, callbacks, config && config.callbacks);
    initPharaohVoice();

    if (!canUseSpeechRecognition) {
      return { supported: false, recognition: false, synthesis: canUseSpeechSynthesis };
    }

    recognition = new SpeechRecognition();
    recognition.lang = config && config.lang ? config.lang : "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    return {
      supported: true,
      recognition: true,
      synthesis: canUseSpeechSynthesis,
      startListening: (pauseMs) => startListening(pauseMs),
      stopListening,
      speak,
      isListening: () => isListening,
      setCallbacks: (c) => { callbacks = Object.assign({}, callbacks, c); },
    };
  }

  global.PharosVoice = {
    init,
    canUseSpeechRecognition: () => canUseSpeechRecognition,
    canUseSpeechSynthesis: () => canUseSpeechSynthesis,
  };
})(typeof window !== "undefined" ? window : this);
