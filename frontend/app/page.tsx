
"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type ScanResult = {
  status: string;
  prediction: string;
  scores: {
    untreated: number;
    starch_treated: number;
  };
  experimental: boolean;
  warning: string;
  preprocessing_note?: string;
};

export default function Home() {
  const [ambient, setAmbient] = useState<File | null>(null);
  const [flash, setFlash] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ScanResult | null>(null);

  const ready = Boolean(ambient && flash && video);

  async function scanPaneer() {
    if (!ambient || !flash || !video) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("ambient_image", ambient);
    formData.append("flash_image", flash);
    formData.append("video", video);

    try {
      const response = await fetch(`${API_URL}/scan`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The scan failed. Please try again."
        );
      }

      setResult(data as ScanResult);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to AnalogueNet."
      );
    } finally {
      setLoading(false);
    }
  }

  function FilePicker({
    title,
    description,
    accept,
    file,
    onChange,
  }: {
    title: string;
    description: string;
    accept: string;
    file: File | null;
    onChange: (file: File | null) => void;
  }) {
    return (
      <label className="block cursor-pointer rounded-2xl border border-stone-200 bg-white p-5 transition hover:border-emerald-500">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-semibold text-stone-900">{title}</h3>
          <span className="text-2xl text-emerald-700">＋</span>
        </div>
        <p className="text-sm text-stone-500">{description}</p>
        <p className="mt-3 truncate text-sm font-medium text-emerald-800">
          {file ? file.name : "Choose file"}
        </p>
        <input
          type="file"
          accept={accept}
          capture={accept.startsWith("image") ? "environment" : undefined}
          className="sr-only"
          onChange={(event) => onChange(event.target.files?.[0] ?? null)}
        />
      </label>
    );
  }

  function formatLabel(value: string) {
    return value === "starch_treated"
      ? "Starch-treated"
      : value === "untreated"
        ? "Untreated"
        : value;
  }

  return (
    <main className="min-h-screen bg-[#f7f8f3] text-stone-900">
      <header className="border-b border-stone-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-800 text-xl text-white">
              A
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">AnalogueNet</h1>
              <p className="text-xs text-stone-500">
                Paneer research prototype
              </p>
            </div>
          </div>
          <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-800">
            Experimental
          </span>
        </div>
      </header>

      <section className="mx-auto max-w-3xl px-5 py-12 sm:py-16">
        <p className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-emerald-800">
          Non-destructive food research
        </p>
        <h2 className="max-w-2xl text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
          Get to know your paneer.
        </h2>
        <p className="mt-5 max-w-xl leading-7 text-stone-600">
          Capture a few views of your paneer sample. AnalogueNet analyses the
          images and pressing motion to generate an experimental classification.
        </p>

        <div className="mt-9 rounded-3xl border border-stone-200 bg-white p-5 shadow-sm sm:p-7">
          <div className="mb-6">
            <h3 className="text-xl font-semibold">Scan your sample</h3>
            <p className="mt-2 text-sm leading-6 text-stone-500">
              Use the same paneer piece for all three captures. Keep it still
              between photos and record a short pressing video.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <FilePicker
              title="1. Ambient photo"
              description="Take a photo without flash."
              accept="image/*"
              file={ambient}
              onChange={setAmbient}
            />
            <FilePicker
              title="2. Flash photo"
              description="Photograph the same piece with flash."
              accept="image/*"
              file={flash}
              onChange={setFlash}
            />
            <div className="sm:col-span-2">
              <FilePicker
                title="3. Pressing video"
                description="Record a short video while pressing gently with a fork."
                accept="video/*"
                file={video}
                onChange={setVideo}
              />
            </div>
          </div>

          <button
            disabled={!ready || loading}
            onClick={scanPaneer}
            className="mt-6 w-full rounded-xl bg-emerald-800 px-5 py-4 font-semibold text-white transition hover:bg-emerald-900 disabled:cursor-not-allowed disabled:bg-stone-300"
          >
            {loading
              ? "Analysing sample..."
              : ready
                ? "Scan Paneer"
                : "Add all 3 captures to continue"}
          </button>

          {loading && (
            <p role="status" className="mt-4 text-center text-sm text-stone-500">
              Aligning photos and analysing the pressing video. This may take
              a little while.
            </p>
          )}

          {error && (
            <div
              role="alert"
              className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-800"
            >
              <strong>Scan unsuccessful.</strong>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <section
              aria-live="polite"
              className="mt-6 rounded-2xl border border-emerald-200 bg-emerald-50 p-5"
            >
              <p className="text-sm font-medium text-emerald-800">
                Experimental model result
              </p>
              <h3 className="mt-2 text-2xl font-bold text-stone-900">
                {formatLabel(result.prediction)}
              </h3>

              <p className="mt-1 text-sm text-stone-600">
                Model scores — not calibrated probabilities of food quality.
              </p>

              {(
                [
                  ["Untreated", result.scores.untreated],
                  ["Starch-treated", result.scores.starch_treated],
                ] as const
              ).map(([label, score]) => (
                <div key={label} className="mt-4">
                  <div className="mb-1 flex justify-between text-sm">
                    <span>{label}</span>
                    <span className="font-semibold">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white">
                    <div
                      className="h-full rounded-full bg-emerald-700"
                      style={{
                        width: `${Math.max(0, Math.min(1, score)) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ))}

              <p className="mt-5 text-sm leading-6 text-amber-900">
                {result.warning}
              </p>

              {result.preprocessing_note && (
                <p className="mt-2 text-xs leading-5 text-stone-600">
                  {result.preprocessing_note}
                </p>
              )}
            </section>
          )}
        </div>

        <p className="mx-auto mt-6 max-w-xl text-center text-xs leading-5 text-stone-500">
          Research prototype only. This classification is not a food-safety
          test and cannot establish whether paneer is safe to eat.
        </p>
      </section>

      <footer className="border-t border-stone-200 px-5 py-6 text-center text-xs text-stone-500">
        AnalogueNet · Multimodal paneer research
      </footer>
    </main>
  );
}
