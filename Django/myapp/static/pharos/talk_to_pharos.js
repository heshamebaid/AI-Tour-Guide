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
  const serviceStatusEl = document.getElementById("service-status");

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const canUseSpeechRecognition = Boolean(SpeechRecognition);
  const canUseSpeechSynthesis = "speechSynthesis" in window;

  let recognition = null;
  let isListening = false;
  let personas = [];
  let activePersona = null;
  let history = [];
  let isSending = false;
  let serviceHealthStatus = null;
  let voiceToVoiceMode = false;
  let pharaohVoice = null;

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

  const findBestPharaohVoice = () => {
    if (!canUseSpeechSynthesis) return null;
    
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;
    
    // Prefer deep, male voices for pharaoh
    // Look for voices with keywords: "male", "deep", "low", or specific voice names
    const preferredKeywords = ["male", "deep", "low", "david", "daniel", "james", "rich"];
    
    // First, try to find a voice with preferred keywords
    for (const voice of voices) {
      const nameLower = voice.name.toLowerCase();
      const langMatch = voice.lang.startsWith("en");
      
      if (langMatch && preferredKeywords.some(keyword => nameLower.includes(keyword))) {
        return voice;
      }
    }
    
    // If no preferred voice found, find the deepest male voice
    // Filter English voices and prefer lower pitch voices
    const englishVoices = voices.filter(v => v.lang.startsWith("en"));
    if (englishVoices.length === 0) return voices[0]; // Fallback to first voice
    
    // Prefer voices that don't sound female (avoid names like "susan", "karen", "zira", etc.)
    const femaleKeywords = ["susan", "karen", "zira", "samantha", "victoria", "kate", "sarah"];
    const maleVoices = englishVoices.filter(v => {
      const nameLower = v.name.toLowerCase();
      return !femaleKeywords.some(keyword => nameLower.includes(keyword));
    });
    
    return maleVoices.length > 0 ? maleVoices[0] : englishVoices[0];
  };

  const initializePharaohVoice = () => {
    if (!canUseSpeechSynthesis) return;
    
    // Load voices (may need to wait for voices to be loaded)
    if (window.speechSynthesis.getVoices().length === 0) {
      window.speechSynthesis.addEventListener("voiceschanged", () => {
        pharaohVoice = findBestPharaohVoice();
      }, { once: true });
    } else {
      pharaohVoice = findBestPharaohVoice();
    }
  };

  const speakAsPharaoh = (text) => {
    if (!canUseSpeechSynthesis || !text) return;
    
    try {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();
      
      // Ensure we have a voice selected
      if (!pharaohVoice) {
        pharaohVoice = findBestPharaohVoice();
      }
      
      // Find the last pharaoh message and mark it as speaking
      const pharaohMessages = timelineEl.querySelectorAll('.message.pharaoh');
      const lastPharaohMessage = pharaohMessages[pharaohMessages.length - 1];
      
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Enhanced voice settings for regal, great pharaoh sound
      utterance.rate = 0.85;  // Slower, more deliberate (was 1.0)
      utterance.pitch = 0.7;   // Lower pitch for deeper, more authoritative voice (was 0.85)
      utterance.volume = 0.95; // Slightly louder for presence (was 0.9)
      utterance.lang = "en-US";
      
      // Use the best available voice
      if (pharaohVoice) {
        utterance.voice = pharaohVoice;
      }
      
      // Visual feedback: mark message as speaking
      utterance.onstart = () => {
        if (lastPharaohMessage) {
          lastPharaohMessage.classList.add('speaking');
        }
      };
      
      // Auto-restart listening after pharaoh finishes speaking (voice-to-voice mode)
      utterance.onend = () => {
        // Remove speaking visual feedback
        if (lastPharaohMessage) {
          lastPharaohMessage.classList.remove('speaking');
        }
        
        // Small delay before restarting to avoid overlap
        setTimeout(() => {
          // Only auto-restart if we're in voice-to-voice mode and not sending
          if (voiceToVoiceMode && isListening === false && !isSending && activePersona) {
            // Check if speech synthesis is not speaking
            if (!window.speechSynthesis.speaking) {
              startListening();
            }
          }
        }, 300);
      };
      
      utterance.onerror = (event) => {
        console.warn("Speech synthesis error:", event.error);
        if (lastPharaohMessage) {
          lastPharaohMessage.classList.remove('speaking');
        }
      };
      
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.warn("Speech synthesis failed", error);
    }
  };

  const sendMessage = async () => {
    if (isSending || !activePersona) {
      if (!activePersona) {
        appendMessage("pharaoh", "⚠️ Please select a Pharaoh first.");
      }
      return;
    }
    const text = inputEl.value.trim();
    if (!text) return;

    // Stop listening if active
    if (isListening) {
      stopListening();
    }

    appendMessage("user", text);
    history.push({ speaker: "user", content: text });
    inputEl.value = "";
    isSending = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "Sending...";

    // Check service health before sending
    const isHealthy = await checkServiceHealth();
    if (!isHealthy) {
      appendMessage("pharaoh", "⚠️ Service is not ready. Please wait a moment and try again.");
      isSending = false;
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
      return;
    }

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
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Service error (${response.status})`);
      }

      const payload = await response.json();
      const answer = payload.answer?.trim() || "The Pharaoh remains silent...";
      appendMessage("pharaoh", answer);
      history.push({ speaker: "pharaoh", content: answer });
      
      // Always speak the pharaoh's response
      speakAsPharaoh(answer);
      
      // In voice-to-voice mode, listening will auto-restart after pharaoh speaks
      // (handled in speakAsPharaoh's onend callback)
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

  const updateServiceStatus = (isHealthy) => {
    if (!serviceStatusEl) return;
    if (isHealthy) {
      serviceStatusEl.textContent = "🟢 Service Ready";
      serviceStatusEl.className = "service-status ready";
    } else {
      serviceStatusEl.textContent = "🔴 Service Unavailable";
      serviceStatusEl.className = "service-status unavailable";
    }
  };

  const checkServiceHealth = async () => {
    try {
      const response = await fetch(`${serviceUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        serviceHealthStatus = data.status === "healthy";
        updateServiceStatus(serviceHealthStatus);
        return serviceHealthStatus;
      } else {
        serviceHealthStatus = false;
        updateServiceStatus(false);
        return false;
      }
    } catch (error) {
      console.warn("Health check failed:", error);
      serviceHealthStatus = false;
      updateServiceStatus(false);
      return false;
    }
  };

  const bootstrap = async () => {
    // Check service health first
    const isHealthy = await checkServiceHealth();
    if (!isHealthy) {
      personaListEl.innerHTML = `<p class="error">⚠️ Service is not ready. Please wait a moment and refresh.</p>`;
      // Retry after 3 seconds
      setTimeout(bootstrap, 3000);
      return;
    }

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
    voiceToVoiceMode = false;
    micBtn.setAttribute("aria-pressed", "false");
    micBtn.textContent = "🎙️ Start Voice Conversation";
    micBtn.classList.remove("listening");
    
    // Clear any pending auto-send timeout
    if (window.autoSendTimeout) {
      clearTimeout(window.autoSendTimeout);
      window.autoSendTimeout = null;
    }
    
    // Cancel any ongoing speech
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
  };

  const startListening = () => {
    if (!recognition || !activePersona) {
      if (!activePersona) {
        appendMessage("pharaoh", "⚠️ Please select a Pharaoh first.");
      }
      return;
    }
    if (isListening) {
      stopListening();
      voiceToVoiceMode = false;
      return;
    }
    try {
      // Enable voice-to-voice mode when starting listening
      voiceToVoiceMode = true;
      
      // Clear input before starting
      inputEl.value = "";
      recognition.start();
      isListening = true;
      micBtn.setAttribute("aria-pressed", "true");
      micBtn.textContent = "🛑 Stop Voice Conversation";
      micBtn.classList.add("listening");
      
      // Add visual indicator (only if conversation is empty or last message wasn't a listening indicator)
      const lastMessage = timelineEl.lastElementChild;
      if (!lastMessage || !lastMessage.textContent.includes("🎙️")) {
        appendMessage("pharaoh", "🎙️ Voice conversation active. Speak naturally - I'll respond with my voice.");
      }
    } catch (error) {
      appendMessage("pharaoh", `⚠️ Microphone error: ${error.message}`);
      stopListening();
      voiceToVoiceMode = false;
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
    recognition.interimResults = true; // Show interim results
    recognition.continuous = true; // Keep listening until stopped
    recognition.maxAlternatives = 1;

    recognition.addEventListener("result", (event) => {
      let interimTranscript = "";
      let finalTranscript = "";
      
      // Separate interim and final results
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + " ";
        } else {
          interimTranscript += transcript;
        }
      }
      
      // Update input with current transcript (final + interim)
      const currentText = (finalTranscript + interimTranscript).trim();
      if (currentText) {
        inputEl.value = currentText;
      }
      
      // If we have final results, auto-send after a brief pause
      if (finalTranscript.trim()) {
        // Clear any existing timeout
        if (window.autoSendTimeout) {
          clearTimeout(window.autoSendTimeout);
        }
        
        // In voice-to-voice mode, use shorter pause for more natural conversation
        const pauseDuration = voiceToVoiceMode ? 800 : 1000;
        
        // Auto-send after pause of no new final results
        window.autoSendTimeout = setTimeout(() => {
          const textToSend = inputEl.value.trim();
          if (textToSend && activePersona && !isSending && isListening) {
            // Remove the listening indicator message if present
            const lastMessage = timelineEl.lastElementChild;
            if (lastMessage && (lastMessage.textContent.includes("🎙️") || 
                                lastMessage.textContent.includes("Listening"))) {
              lastMessage.remove();
            }
            
            // Temporarily stop listening while sending (will restart after pharaoh speaks)
            const wasInVoiceMode = voiceToVoiceMode;
            if (wasInVoiceMode) {
              recognition.stop();
              isListening = false;
            }
            
            // Send message - listening will restart automatically after pharaoh speaks
            // via the speakAsPharaoh onend callback
            sendMessage();
          }
          window.autoSendTimeout = null;
        }, pauseDuration);
      }
    });

    recognition.addEventListener("end", () => {
      // In voice-to-voice mode, we'll restart listening after pharaoh speaks
      // (handled by speakAsPharaoh's onend callback)
      // Only auto-restart here if we're still supposed to be listening
      // and not waiting for pharaoh to respond
      if (isListening && recognition.continuous && !isSending && voiceToVoiceMode) {
        // Check if pharaoh is currently speaking - if so, don't restart yet
        if (!window.speechSynthesis.speaking) {
          try {
            recognition.start();
          } catch (error) {
            // Recognition might already be starting, ignore the error
            console.debug("Recognition restart:", error.message);
          }
        }
      } else if (isListening && !voiceToVoiceMode) {
        // Not in voice-to-voice mode, restart normally
        try {
          recognition.start();
        } catch (error) {
          console.debug("Recognition restart:", error.message);
        }
      }
    });

    recognition.addEventListener("error", (event) => {
      let errorMessage = "Voice recognition error";
      switch (event.error) {
        case "no-speech":
          errorMessage = "No speech detected. Please try again.";
          break;
        case "audio-capture":
          errorMessage = "Microphone not found. Please check your microphone.";
          break;
        case "not-allowed":
          errorMessage = "Microphone permission denied. Please allow microphone access.";
          break;
        case "network":
          errorMessage = "Network error during voice recognition.";
          break;
        default:
          errorMessage = `Voice error: ${event.error}`;
      }
      appendMessage("pharaoh", `⚠️ ${errorMessage}`);
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

  // Periodic health check every 30 seconds
  const startHealthCheckInterval = () => {
    setInterval(async () => {
      await checkServiceHealth();
    }, 30000);
  };

  bootstrap();
  initVoiceFeatures();
  initializePharaohVoice();
  startHealthCheckInterval();
})();

