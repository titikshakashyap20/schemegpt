"use client";

import { useState, useEffect, useRef } from "react";

interface SourceMeta {
  source?: string;
  chunk_index?: number;
  similarity_score?: number;
  [key: string]: any;
}

interface BackendResponse {
  question: string;
  answer: string;
  sources: SourceMeta[];
  detected_scheme?: string | null;
  confidence?: number | null;
}

interface Message {
  sender: "user" | "ai";
  text: string;
  sources?: SourceMeta[];
  detectedScheme?: string | null;
  confidence?: number | null;
}

const BACKEND_URL = "http://127.0.0.1:8001/query";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    setErrorMsg(null);

    const userMessage: Message = {
      sender: "user",
      text: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const questionToSend = input.trim();
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: questionToSend }),
      });

      if (!res.ok) {
        throw new Error(`Backend error: ${res.status}`);
      }

      const data: BackendResponse = await res.json();

      const aiMessage: Message = {
        sender: "ai",
        text: data.answer || "No answer received.",
        sources: data.sources || [],
        detectedScheme: data.detected_scheme ?? null,
        confidence: data.confidence ?? null,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error(err);
      setErrorMsg("Backend not reachable. Make sure FastAPI is running on port 8001.");
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: "⚠️ I couldn't reach the backend. Please check if the server is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      sendMessage();
    }
  }

  // Aggregate sources for pretty display
  function summarizeSources(sources: SourceMeta[] | undefined) {
    if (!sources || sources.length === 0) return [];
    const counts: Record<string, { count: number; maxScore: number }> = {};
    for (const src of sources) {
      const name = (src.source || "Unknown").toString();
      const score = typeof src.similarity_score === "number" ? src.similarity_score : 0;
      if (!counts[name]) {
        counts[name] = { count: 0, maxScore: 0 };
      }
      counts[name].count += 1;
      if (score > counts[name].maxScore) {
        counts[name].maxScore = score;
      }
    }
    return Object.entries(counts).map(([name, info]) => ({
      name,
      count: info.count,
      maxScore: info.maxScore,
    }));
  }

  function formatSchemeKey(key?: string | null) {
    if (!key) return null;
    const k = key.toLowerCase();
    if (k === "pmjdy") return "Pradhan Mantri Jan Dhan Yojana (PMJDY)";
    if (k === "nsp") return "National Scholarship Portal (NSP)";
    if (k === "ayushman") return "Ayushman Bharat – PM-JAY";
    if (k === "pmay-g") return "Pradhan Mantri Awas Yojana – Gramin (PMAY-G)";
    if (k === "pmay-u") return "Pradhan Mantri Awas Yojana – Urban (PMAY-U)";
    if (k === "mudra") return "Pradhan Mantri Mudra Yojana (MUDRA)";
    return key.toUpperCase();
  }

  function renderConfidenceBar(confidence?: number | null) {
    if (confidence == null || isNaN(confidence)) return null;
    const pct = Math.round(confidence * 100);
    const width = Math.min(Math.max(pct, 5), 100); // avoid 0-width
    let color = "bg-emerald-400";
    if (pct < 40) color = "bg-red-400";
    else if (pct < 70) color = "bg-amber-400";

    return (
      <div className="mt-3">
        <div className="flex items-center justify-between text-xs text-slate-300 mb-1">
          <span>Answer confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="w-full h-2 rounded-full bg-slate-800/70 overflow-hidden">
          <div
            className={`h-full ${color} transition-all duration-500`}
            style={{ width: `${width}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-50 flex flex-col items-center px-4 py-8">
      {/* Glass Hero */}
      <header className="w-full max-w-4xl mb-6 flex flex-col items-center justify-center text-center">
        <div className="px-4 py-2 rounded-full border border-slate-700/60 bg-slate-900/40 backdrop-blur-md text-xs uppercase tracking-[0.2em] text-slate-400 mb-3">
          India · Government Schemes · AI Assistant
        </div>
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-2">
          Scheme<span className="bg-gradient-to-r from-sky-400 to-emerald-400 bg-clip-text text-transparent">GPT</span>
        </h1>
        <p className="text-slate-300 max-w-2xl text-sm md:text-base">
          A smart portal to explore and understand government schemes across India.
        </p>
      </header>

      {/* Main Glass Card */}
      <main className="w-full max-w-4xl flex-1 flex flex-col gap-4">
        {/* Error banner */}
        {errorMsg && (
          <div className="mb-2 rounded-xl bg-red-500/10 border border-red-500/40 px-4 py-2 text-xs text-red-200">
            {errorMsg}
          </div>
        )}

        <div className="relative flex-1 rounded-3xl border border-slate-800/80 bg-slate-900/60 backdrop-blur-xl shadow-[0_0_40px_rgba(15,23,42,0.8)] flex flex-col overflow-hidden">
          {/* Chat area */}
          <div
            ref={chatRef}
            className="flex-1 overflow-y-auto px-5 pt-5 pb-24 space-y-4 scrollbar-thin scrollbar-thumb-slate-700/80 scrollbar-track-transparent"
          >
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-sm text-slate-400 gap-1">
                <p className="text-slate-300">Try asking:</p>
                <p>“What is the eligibility for PMJDY?”</p>
                <p>“Who can get Ayushman Bharat benefits?”</p>
                <p>“Explain PMAY-G eligibility in simple terms.”</p>
              </div>
            )}

            {messages.map((msg, index) => {
              const isUser = msg.sender === "user";
              const schemeName = !isUser ? formatSchemeKey(msg.detectedScheme) : null;
              const sourceSummary = summarizeSources(msg.sources);

              return (
                <div
                  key={index}
                  className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-md ${
                      isUser
                        ? "bg-sky-500/90 text-white rounded-br-sm"
                        : "bg-slate-800/80 text-slate-50 border border-slate-700/60 rounded-bl-sm"
                    }`}
                  >
                    {/* Scheme badge + confidence for AI messages */}
                    {!isUser && (schemeName || msg.confidence != null) && (
                      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                        {schemeName && (
                          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-sky-500/10 border border-sky-400/40 text-sky-100">
                            <span>📘</span>
                            <span>{schemeName}</span>
                          </span>
                        )}
                        {msg.confidence != null && (
                          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-400/40 text-emerald-100">
                            <span>✅</span>
                            <span>Conf: {Math.round((msg.confidence ?? 0) * 100)}%</span>
                          </span>
                        )}
                      </div>
                    )}

                    <p className="whitespace-pre-line leading-relaxed">{msg.text}</p>

                    {/* Confidence bar under AI answer */}
                    {!isUser && renderConfidenceBar(msg.confidence)}

                    {/* Sources */}
                    {!isUser && sourceSummary.length > 0 && (
                      <div className="mt-3 border-t border-slate-700/60 pt-2">
                        <p className="text-xs text-slate-300 mb-1">Sources used:</p>
                        <div className="flex flex-wrap gap-1">
                          {sourceSummary.map((s, i) => (
                            <span
                              key={i}
                              className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-slate-900/80 border border-slate-700/70 text-[11px] text-slate-200"
                            >
                              📄 {s.name}{" "}
                              <span className="text-slate-400">
                                ({s.count} snippet{s.count > 1 ? "s" : ""})
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Typing indicator */}
            {loading && (
              <div className="flex justify-start mt-2">
                <div className="inline-flex items-center gap-2 px-3 py-2 rounded-2xl bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300">
                  <span>SchemeGPT is thinking</span>
                  <span className="flex gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse"></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-pulse [animation-delay:150ms]"></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 animate-pulse [animation-delay:300ms]"></span>
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Input bar (glass) */}
          <div className="absolute bottom-0 left-0 right-0 border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-xl px-4 py-3">
            <div className="flex gap-3 items-center">
              <input
                type="text"
                placeholder="Ask about any government scheme…"
                className="flex-1 bg-slate-900/80 border border-slate-700/80 rounded-2xl px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 outline-none focus:ring-2 focus:ring-sky-500/70 focus:border-sky-500/70"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button
                onClick={sendMessage}
                disabled={loading}
                className="px-5 py-2.5 rounded-2xl bg-gradient-to-r from-sky-500 to-emerald-400 text-slate-950 text-sm font-semibold shadow hover:from-sky-400 hover:to-emerald-300 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

