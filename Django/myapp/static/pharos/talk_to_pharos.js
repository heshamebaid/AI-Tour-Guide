(() => {
  const appEl = document.getElementById("pharos-app");
  if (!appEl) return;

  const serviceUrl = appEl.dataset.serviceUrl?.replace(/\/$/, "") || "http://localhost:8050";

  const personaListEl = document.getElementById("persona-list");
  const timelineEl = document.getElementById("conversation-timeline");
  const inputEl = document.getElementById("traveler-input");
  const sendBtn = document.getElementById("send-message");
  const clearBtn = document.getElementById("clear-history");
  const micBtn = document.getElementById("mic-toggle");
  const activeNameEl = document.getElementById("active-pharaoh-name");
  const activeEraEl = document.getElementById("active-pharaoh-era");

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const canUseSpeechRecognition = Boolean(SpeechRecognition);
  const canUseSpeechSynthesis = "speechSynthesis" in window;

  let recognition = null;
  let isListening = false;
  let personas = [];
  let activePersona = null;
  let history = [];
  let isSending = false;

  const renderPersonas = () => {
    personaListEl.innerHTML = "";
    personas.forEach((persona) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = `persona-card${activePersona?.id === persona.id ? " active" : ""}`;
      card.dataset.personaId = persona.id;
      card.innerHTML = `
        <div class="avatar">${persona.display_name[0] || "?"}</div>
        <div>
          <h3>${persona.display_name}</h3>
          <p>${persona.short_bio}</p>
        </div>
      `;
      card.addEventListener("click", () => selectPersona(persona.id));
      personaListEl.appendChild(card);
    });
  };

  const selectPersona = (personaId) => {
    const persona = personas.find((p) => p.id === personaId);
    if (!persona) return;
    activePersona = persona;
    history = [];
    timelineEl.innerHTML = `
      <div class="empty-state">
        <p>You are now connected with ${persona.display_name}. Greet them to begin.</p>
      </div>
    `;
    activeNameEl.textContent = persona.display_name;
    activeEraEl.textContent = persona.era;
    renderPersonas();
  };

  const appendMessage = (speaker, text) => {
    if (timelineEl.querySelector(".empty-state")) {
      timelineEl.innerHTML = "";
    }
    const messageEl = document.createElement("div");
    messageEl.className = `message ${speaker}`;
    messageEl.innerHTML = `
      <p class="speaker">${speaker === "user" ? "Traveler" : activePersona?.display_name || "Pharaoh"}</p>
      <p>${text}</p>
    `;
    timelineEl.appendChild(messageEl);
    timelineEl.scrollTop = timelineEl.scrollHeight;
  };

  const speakAsPharaoh = (text) => {
    if (!canUseSpeechSynthesis || !text) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1;
      utterance.pitch = 0.85;
      utterance.volume = 0.9;
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.warn("Speech synthesis failed", error);
    }
  };

  const sendMessage = async () => {
    if (isSending || !activePersona) return;
    const text = inputEl.value.trim();
    if (!text) return;

    appendMessage("user", text);
    history.push({ speaker: "user", content: text });
    inputEl.value = "";
    isSending = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "Sending...";

    try {
      const response = await fetch(`${serviceUrl}/converse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pharaoh_id: activePersona.id,
          user_query: text,
          history,
        }),
      });

      if (!response.ok) {
        throw new Error(`Service error (${response.status})`);
      }

      const payload = await response.json();
      const answer = payload.answer?.trim() || "The Pharaoh remains silent...";
      appendMessage("pharaoh", answer);
      history.push({ speaker: "pharaoh", content: answer });
      speakAsPharaoh(answer);
    } catch (error) {
      appendMessage("pharaoh", `⚠️ ${error.message || "Failed to reach the Pharaoh"}`);
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  };

  const clearConversation = () => {
    history = [];
    timelineEl.innerHTML = `<div class="empty-state"><p>Conversation cleared.</p></div>`;
  };

  const bootstrap = async () => {
    try {
      const response = await fetch(`${serviceUrl}/pharos`);
      if (!response.ok) throw new Error("Failed to load personas");
      personas = await response.json();
      if (!Array.isArray(personas) || personas.length === 0) {
        personaListEl.innerHTML = "<p>No personas available.</p>";
        return;
      }
      renderPersonas();
      selectPersona(personas[0].id);
    } catch (error) {
      personaListEl.innerHTML = `<p class="error">Unable to load personas: ${error.message}</p>`;
    }
  };

  const stopListening = () => {
    if (recognition && isListening) {
      recognition.stop();
    }
    isListening = false;
    micBtn.setAttribute("aria-pressed", "false");
    micBtn.textContent = "🎙️ Start Listening";
  };

  const startListening = () => {
    if (!recognition) return;
    if (isListening) {
      stopListening();
      return;
    }
    try {
      recognition.start();
      isListening = true;
      micBtn.setAttribute("aria-pressed", "true");
      micBtn.textContent = "🛑 Stop Listening";
    } catch (error) {
      appendMessage("pharaoh", `⚠️ Microphone error: ${error.message}`);
    }
  };

  const initVoiceFeatures = () => {
    if (!canUseSpeechRecognition) {
      micBtn.disabled = true;
      micBtn.textContent = "🎙️ Voice not supported";
      micBtn.classList.add("disabled");
      return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.addEventListener("result", (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(" ");
      const appended = `${inputEl.value} ${transcript}`.trim();
      inputEl.value = appended;
    });

    recognition.addEventListener("end", () => {
      if (isListening) {
        stopListening();
      }
    });

    recognition.addEventListener("error", (event) => {
      appendMessage("pharaoh", `⚠️ Voice error: ${event.error}`);
      stopListening();
    });

    micBtn.addEventListener("click", startListening);
  };

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  clearBtn.addEventListener("click", clearConversation);

  bootstrap();
  initVoiceFeatures();
})();

