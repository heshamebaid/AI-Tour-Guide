/**
 * Talk to Pharos - Main app (Voice-to-Voice first)
 * Depends on pharos-voice.js. Handles: API, personas, UI, and wiring the voice loop.
 */
(function () {
  "use strict";

  const appEl = document.getElementById("pharos-app");
  if (!appEl) return;

  const serviceUrl = (appEl.dataset.serviceUrl || "http://localhost:8050").replace(/\/$/, "");
  const personaListEl = document.getElementById("persona-list");
  const timelineEl = document.getElementById("conversation-timeline");
  const inputEl = document.getElementById("traveler-input");
  const sendBtn = document.getElementById("send-message");
  const clearBtn = document.getElementById("clear-history");
  const micBtn = document.getElementById("mic-toggle");
  const activeNameEl = document.getElementById("active-pharaoh-name");
  const activeEraEl = document.getElementById("active-pharaoh-era");
  const serviceStatusEl = document.getElementById("service-status");

  let personas = [];
  let activePersona = null;
  let history = [];
  let isSending = false;
  let voiceToVoiceMode = false;
  let voiceApi = null;

  // ---------- API ----------
  async function checkHealth() {
    try {
      const res = await fetch(`${serviceUrl}/health`);
      const data = res.ok ? await res.json() : {};
      const ok = data.status === "healthy";
      updateServiceStatus(ok);
      return ok;
    } catch (e) {
      updateServiceStatus(false);
      return false;
    }
  }

  async function fetchPersonas() {
    const res = await fetch(`${serviceUrl}/pharos`);
    if (!res.ok) throw new Error("Failed to load personas");
    const list = await res.json();
    return Array.isArray(list) ? list : [];
  }

  async function converse(userText) {
    const res = await fetch(`${serviceUrl}/converse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pharaoh_id: activePersona.id,
        user_query: userText,
        history,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Service error");
    }
    const payload = await res.json();
    return (payload.answer || "").trim() || "The Pharaoh remains silent...";
  }

  // ---------- UI ----------
  function updateServiceStatus(ok) {
    if (!serviceStatusEl) return;
    serviceStatusEl.textContent = ok ? "🟢 Service Ready" : "🔴 Service Unavailable";
    serviceStatusEl.className = "service-status " + (ok ? "ready" : "unavailable");
  }

  function renderPersonas() {
    personaListEl.innerHTML = "";
    personas.forEach((p) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "persona-card" + (activePersona && activePersona.id === p.id ? " active" : "");
      card.dataset.personaId = p.id;
      card.innerHTML = `<div class="avatar">${(p.display_name || "?")[0]}</div><div><h3>${p.display_name}</h3><p>${p.short_bio}</p></div>`;
      card.addEventListener("click", () => selectPersona(p.id));
      personaListEl.appendChild(card);
    });
  }

  function selectPersona(personaId) {
    const p = personas.find((x) => x.id === personaId);
    if (!p) return;
    activePersona = p;
    history = [];
    timelineEl.innerHTML = `<div class="empty-state"><p>You are now connected with ${p.display_name}. Greet them to begin.</p></div>`;
    activeNameEl.textContent = p.display_name;
    activeEraEl.textContent = p.era || "";
    renderPersonas();
  }

  function appendMessage(speaker, text) {
    if (timelineEl.querySelector(".empty-state")) timelineEl.innerHTML = "";
    const msg = document.createElement("div");
    msg.className = "message " + speaker;
    const label = speaker === "user" ? "Traveler" : (activePersona && activePersona.display_name) || "Pharaoh";
    msg.innerHTML = `<p class="speaker">${label}</p><p>${text}</p>`;
    timelineEl.appendChild(msg);
    timelineEl.scrollTop = timelineEl.scrollHeight;
    return msg;
  }

  function removeLastMessageIfListeningIndicator() {
    const last = timelineEl.lastElementChild;
    if (last && (last.textContent.includes("🎙️") || last.textContent.includes("Listening"))) last.remove();
  }

  function setMicButton(listening) {
    micBtn.setAttribute("aria-pressed", listening ? "true" : "false");
    micBtn.textContent = listening ? "🛑 Stop Voice Conversation" : "🎙️ Start Voice Conversation";
    micBtn.classList.toggle("listening", listening);
  }

  // ---------- Send message (text or from voice) ----------
  async function sendMessage(optionalText) {
    const text = (optionalText != null ? optionalText : inputEl.value || "").trim();
    if (!text || !activePersona) {
      if (!activePersona) appendMessage("pharaoh", "⚠️ Please select a Pharaoh first.");
      return null;
    }

    if (voiceApi && voiceApi.isListening()) voiceApi.stopListening();

    appendMessage("user", text);
    history.push({ speaker: "user", content: text });
    inputEl.value = "";
    isSending = true;
    sendBtn.disabled = true;
    if (sendBtn.textContent === "Send") sendBtn.textContent = "Sending...";

    try {
      const healthy = await checkHealth();
      if (!healthy) {
        appendMessage("pharaoh", "⚠️ Service is not ready. Please wait and try again.");
        return null;
      }
      const answer = await converse(text);
      appendMessage("pharaoh", answer);
      history.push({ speaker: "pharaoh", content: answer });
      if (voiceApi && voiceApi.speak) {
        voiceApi.speak(answer, {
          onStart: () => markLastPharaohSpeaking(true),
          onEnd: () => markLastPharaohSpeaking(false),
        });
      }
      return answer;
    } catch (err) {
      appendMessage("pharaoh", "⚠️ " + (err.message || "Failed to reach the Pharaoh"));
      return null;
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  }

  // ---------- Voice-to-voice loop ----------
  function markLastPharaohSpeaking(speaking) {
    const pharaohMsgs = timelineEl.querySelectorAll(".message.pharaoh");
    if (pharaohMsgs.length) pharaohMsgs[pharaohMsgs.length - 1].classList.toggle("speaking", speaking);
  }

  function onFinalTranscript(text) {
    removeLastMessageIfListeningIndicator();
    sendMessage(text).then((answer) => {
      if (answer && voiceApi && voiceToVoiceMode) {
        voiceApi.speak(answer, {
          onStart: () => markLastPharaohSpeaking(true),
          onEnd: () => {
            markLastPharaohSpeaking(false);
            if (voiceToVoiceMode && !isSending && activePersona && voiceApi) setTimeout(() => voiceApi.startListening(800), 300);
          },
        });
      }
    });
  }

  function startVoiceConversation() {
    if (!activePersona) {
      appendMessage("pharaoh", "⚠️ Please select a Pharaoh first.");
      return;
    }
    if (!voiceApi || !voiceApi.supported) return;
    voiceToVoiceMode = true;
    inputEl.value = "";
    appendMessage("pharaoh", "🎙️ Voice conversation active. Speak naturally — I'll respond with my voice.");
    voiceApi.startListening(800);
  }

  function stopVoiceConversation() {
    voiceToVoiceMode = false;
    if (voiceApi) voiceApi.stopListening();
  }

  function toggleMic() {
    if (voiceApi && voiceApi.isListening()) {
      stopVoiceConversation();
      return;
    }
    startVoiceConversation();
  }

  // ---------- Bootstrap ----------
  async function bootstrap() {
    const healthy = await checkHealth();
    if (!healthy) {
      personaListEl.innerHTML = "<p class=\"error\">⚠️ Service not ready. Retrying...</p>";
      setTimeout(bootstrap, 3000);
      return;
    }
    try {
      personas = await fetchPersonas();
      if (!personas.length) {
        personaListEl.innerHTML = "<p>No personas available.</p>";
        return;
      }
      renderPersonas();
      selectPersona(personas[0].id);
    } catch (e) {
      personaListEl.innerHTML = "<p class=\"error\">" + (e.message || "Unable to load personas") + "</p>";
    }
  }

  function initVoice() {
    voiceApi = PharosVoice.init({
      lang: "en-US",
      callbacks: {
        onInterimTranscript: (text) => { inputEl.value = text; },
        onFinalTranscript,
        onSpeakEnd: () => {
          if (voiceToVoiceMode && !isSending && activePersona && voiceApi) voiceApi.startListening(800);
        },
        onError: (source, msg) => appendMessage("pharaoh", "⚠️ " + (typeof msg === "string" ? msg : msg.message || "Voice error")),
        onListeningChange: setMicButton,
      },
    });

    if (!voiceApi.supported) {
      micBtn.disabled = true;
      micBtn.textContent = "🎙️ Voice not supported";
      micBtn.classList.add("disabled");
      return;
    }

    micBtn.addEventListener("click", toggleMic);
  }

  // ---------- Event bindings ----------
  sendBtn.addEventListener("click", () => sendMessage());
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  clearBtn.addEventListener("click", () => {
    history = [];
    timelineEl.innerHTML = "<div class=\"empty-state\"><p>Conversation cleared.</p></div>";
  });

  setInterval(checkHealth, 30000);

  // ---------- Start ----------
  bootstrap();
  initVoice();
})();
