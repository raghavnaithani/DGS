"use client";

import { useMemo, useState } from "react";

type QuestionType = "text" | "choice" | "number";

type IntakeQuestion = {
  id: string;
  text: string;
  type: QuestionType;
  choices?: string[];
  hint?: string | null;
};

type UserIntent = {
  id: string;
  original_prompt: string;
  domain: string;
  horizon_months: number;
  risk_tolerance: number;
  constraints: string[];
  personal_context: string;
  clarified_entities: string[];
  ambiguities_remaining: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const DEFAULT_USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_INTAKE !== "false";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = typeof payload === "string" ? payload : payload?.detail ?? "Request failed";
    throw new Error(message);
  }

  return payload as T;
}

export function IntakeWizard() {
  const [useMock, setUseMock] = useState(DEFAULT_USE_MOCK);
  const [step, setStep] = useState<"prompt" | "questions" | "intent">("prompt");
  const [prompt, setPrompt] = useState("");
  const [questions, setQuestions] = useState<IntakeQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [intent, setIntent] = useState<UserIntent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentQuestion = questions[currentIndex] ?? null;
  const currentAnswer = currentQuestion ? answers[currentQuestion.id] ?? "" : "";
  const progress = useMemo(() => {
    if (step === "prompt" || questions.length === 0) {
      return 0;
    }
    return Math.round(((currentIndex + 1) / questions.length) * 100);
  }, [currentIndex, questions.length, step]);

  const updateAnswer = (value: string) => {
    if (!currentQuestion) {
      return;
    }
    setAnswers((previous) => ({ ...previous, [currentQuestion.id]: value }));
  };

  const startClarify = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt first.");
      return;
    }

    setLoading(true);
    setError(null);
    setIntent(null);

    try {
      const endpoint = useMock ? "/v1/intake/mock-clarify?prompt=test" : "/v1/intake/clarify";
      const payload = useMock
        ? await fetchJson<{ questions: IntakeQuestion[] }>(endpoint)
        : await fetchJson<{ questions: IntakeQuestion[] }>(endpoint, {
            method: "POST",
            body: JSON.stringify({ prompt }),
          });

      setQuestions(payload.questions);
      setAnswers({});
      setCurrentIndex(0);
      setStep("questions");
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to clarify prompt.");
    } finally {
      setLoading(false);
    }
  };

  const goNext = async () => {
    if (!currentQuestion) {
      return;
    }

    if (!currentAnswer.trim()) {
      setError("Please answer the current question before continuing.");
      return;
    }

    setError(null);

    if (currentIndex < questions.length - 1) {
      setCurrentIndex((value) => value + 1);
      return;
    }

    setLoading(true);
    try {
      const endpoint = useMock ? "/v1/intake/mock-build-intent" : "/v1/intake/build-intent";
      const payload = useMock
        ? await fetchJson<UserIntent>(endpoint, {
            method: "POST",
            body: JSON.stringify({ prompt, answers }),
          })
        : await fetchJson<UserIntent>(endpoint, {
            method: "POST",
            body: JSON.stringify({ prompt, answers }),
          });
      setIntent(payload);
      setStep("intent");
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to build intent.");
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    setError(null);
    if (step === "questions" && currentIndex > 0) {
      setCurrentIndex((value) => value - 1);
      return;
    }
    if (step === "questions") {
      setStep("prompt");
    }
  };

  const resetFlow = () => {
    setError(null);
    setQuestions([]);
    setAnswers({});
    setCurrentIndex(0);
    setIntent(null);
    setStep("prompt");
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_20px_60px_rgba(15,23,42,0.12)]">
      <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Phase 2 Intake</p>
            <h2 className="mt-1 text-2xl font-semibold text-slate-900">Smart Intake & Clarification Engine</h2>
          </div>
          <label className="flex items-center gap-3 text-sm font-medium text-slate-700">
            <span>Use mock endpoints</span>
            <button
              type="button"
              onClick={() => setUseMock((value) => !value)}
              className={`relative h-8 w-14 rounded-full border transition ${useMock ? "border-emerald-500 bg-emerald-500" : "border-slate-300 bg-slate-200"}`}
              aria-pressed={useMock}
            >
              <span
                className={`absolute top-1 h-6 w-6 rounded-full bg-white shadow transition ${useMock ? "left-7" : "left-1"}`}
              />
            </button>
          </label>
        </div>
      </div>

      <div className="space-y-6 p-6">
        {step !== "prompt" ? (
          <div>
            <div className="mb-2 flex items-center justify-between text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              <span>Progress</span>
              <span>{progress}%</span>
            </div>
            <div className="h-2 rounded-full bg-slate-100">
              <div className="h-2 rounded-full bg-slate-900 transition-all" style={{ width: `${progress}%` }} />
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div>
        ) : null}

        {step === "prompt" ? (
          <div className="space-y-4">
            <div>
              <label htmlFor="prompt" className="mb-2 block text-sm font-medium text-slate-700">
                What are you trying to decide?
              </label>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                maxLength={2000}
                rows={6}
                placeholder="For example: Should I change my career?"
                className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
              />
              <p className="mt-2 text-xs text-slate-500">Max 2000 characters. A clear prompt helps generate better clarifying questions.</p>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-600">
                {useMock ? "Mock intake is enabled for fast frontend development." : "Real Groq-powered intake is enabled."}
              </p>
              <button
                type="button"
                onClick={startClarify}
                disabled={loading}
                className="inline-flex items-center justify-center rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Starting…" : "Start"}
              </button>
            </div>
          </div>
        ) : null}

        {step === "questions" && currentQuestion ? (
          <div className="space-y-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                Question {currentIndex + 1} of {questions.length}
              </p>
              <h3 className="mt-2 text-xl font-semibold text-slate-900">{currentQuestion.text}</h3>
              {currentQuestion.hint ? <p className="mt-2 text-sm text-slate-600">{currentQuestion.hint}</p> : null}
            </div>

            <div>
              {currentQuestion.type === "choice" ? (
                <select
                  value={currentAnswer}
                  onChange={(event) => updateAnswer(event.target.value)}
                  className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
                >
                  <option value="">Select an option</option>
                  {currentQuestion.choices?.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              ) : null}

              {currentQuestion.type === "number" ? (
                <div className="space-y-3">
                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={currentAnswer || "5"}
                    onChange={(event) => updateAnswer(event.target.value)}
                    className="w-full accent-slate-900"
                  />
                  <div className="flex items-center justify-between text-sm text-slate-600">
                    <span>1</span>
                    <span className="font-semibold text-slate-900">{currentAnswer || "5"}</span>
                    <span>10</span>
                  </div>
                </div>
              ) : null}

              {currentQuestion.type === "text" ? (
                <input
                  type="text"
                  value={currentAnswer}
                  onChange={(event) => updateAnswer(event.target.value)}
                  placeholder="Type your answer here"
                  className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
                />
              ) : null}
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="button"
                onClick={goBack}
                className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
              >
                Back
              </button>
              <button
                type="button"
                onClick={goNext}
                disabled={loading}
                className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Working…" : currentIndex < questions.length - 1 ? "Next" : "Build User Intent"}
              </button>
            </div>
          </div>
        ) : null}

        {step === "intent" && intent ? (
          <div className="space-y-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-600">User Intent</p>
              <h3 className="mt-2 text-2xl font-semibold text-slate-900">Final structured summary</h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <IntentCard label="Domain" value={intent.domain} />
              <IntentCard label="Horizon (months)" value={String(intent.horizon_months)} />
              <IntentCard label="Risk tolerance" value={String(intent.risk_tolerance)} />
              <IntentCard label="Constraints" value={intent.constraints.join(", ") || "None"} />
              <IntentCard label="Personal context" value={intent.personal_context} span={2} />
              <IntentCard label="Clarified entities" value={intent.clarified_entities.join(", ") || "None"} span={2} />
              <IntentCard label="Ambiguities remaining" value={intent.ambiguities_remaining.join(", ") || "None"} span={2} />
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm text-slate-600">
                Intent ID: <span className="font-mono text-slate-900">{intent.id}</span>
              </div>
              <button
                type="button"
                onClick={resetFlow}
                className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
              >
                Start over
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function IntentCard({ label, value, span = 1 }: { label: string; value: string; span?: number }) {
  return (
    <div className={`rounded-2xl border border-slate-200 bg-slate-50 p-4 ${span > 1 ? "md:col-span-2" : ""}`}>
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <div className="mt-2 text-sm leading-6 text-slate-900">{value}</div>
    </div>
  );
}