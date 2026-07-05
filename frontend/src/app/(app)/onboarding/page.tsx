"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiJson } from "@/lib/api";

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [expertiseLevel, setExpertiseLevel] = useState("intermediate");
  const [riskTolerance, setRiskTolerance] = useState(5);
  const [values, setValues] = useState<string[]>([]);
  const [lifeSituation, setLifeSituation] = useState("");

  const valuesOptions = [
    "Financial growth",
    "Work-life balance",
    "Learning",
    "Stability",
    "Impact",
    "Freedom",
    "Health",
    "Family",
  ];

  const handleExpertise = (level: string) => {
    setExpertiseLevel(level);
    setStep(2);
  };

  const toggleValue = (val: string) => {
    if (values.includes(val)) {
      setValues(values.filter((v) => v !== val));
    } else if (values.length < 3) {
      setValues([...values, val]);
    }
  };

  const getRiskLabel = (val: number) => {
    if (val <= 3) return "Very Cautious 🛡️";
    if (val <= 6) return "Balanced ⚖️";
    if (val <= 8) return "Growth-Oriented 🚀";
    return "Bold Opportunist 🔥";
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    try {
      await apiJson("/profile", {
        method: "POST",
        body: JSON.stringify({
          expertise_level: expertiseLevel,
          risk_tolerance: riskTolerance,
          values: values,
          life_situation: lifeSituation,
        }),
      });
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "Failed to save profile");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto py-12 px-4">
      {/* Step dots */}
      <div className="flex justify-center space-x-2 mb-8">
        {[1, 2, 3, 4].map((s) => (
          <div
            key={s}
            className={`h-2 w-2 rounded-full ${s === step ? "bg-blue-600" : s < step ? "bg-blue-300" : "bg-gray-200"}`}
          />
        ))}
      </div>

      {error && <div className="mb-4 text-red-600 bg-red-50 p-3 rounded">{error}</div>}

      {step === 1 && (
        <div>
          <h2 className="text-2xl font-bold mb-6">What is your expertise level?</h2>
          <div className="space-y-4">
            <button onClick={() => handleExpertise("beginner")} className="w-full text-left p-4 border rounded hover:border-blue-500 focus:border-blue-500 bg-white">
              <div className="font-semibold text-lg">Beginner</div>
              <div className="text-gray-600">I'm exploring this for the first time</div>
            </button>
            <button onClick={() => handleExpertise("intermediate")} className="w-full text-left p-4 border rounded hover:border-blue-500 focus:border-blue-500 bg-white">
              <div className="font-semibold text-lg">Intermediate</div>
              <div className="text-gray-600">I have some experience</div>
            </button>
            <button onClick={() => handleExpertise("expert")} className="w-full text-left p-4 border rounded hover:border-blue-500 focus:border-blue-500 bg-white">
              <div className="font-semibold text-lg">Expert</div>
              <div className="text-gray-600">I work in this domain professionally</div>
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div>
          <h2 className="text-2xl font-bold mb-6">What is your risk tolerance?</h2>
          <div className="mb-8 text-center">
            <div className="text-xl font-medium text-blue-700 mb-2">{getRiskLabel(riskTolerance)}</div>
            <div className="text-gray-500 mb-6">Level: {riskTolerance}/10</div>
            <input
              type="range"
              min="1"
              max="10"
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <button onClick={() => setStep(3)} className="w-full py-3 bg-blue-600 text-white rounded font-medium hover:bg-blue-700">Next</button>
        </div>
      )}

      {step === 3 && (
        <div>
          <h2 className="text-2xl font-bold mb-2">Select your core values</h2>
          <p className="text-gray-600 mb-6">Choose up to 3 values that matter most to you.</p>
          <div className="flex flex-wrap gap-3 mb-6">
            {valuesOptions.map((val) => {
              const isSelected = values.includes(val);
              const isDisabled = !isSelected && values.length >= 3;
              return (
                <button
                  key={val}
                  onClick={() => toggleValue(val)}
                  disabled={isDisabled}
                  className={`px-4 py-2 rounded-full border ${isSelected ? "bg-blue-100 border-blue-500 text-blue-700" : isDisabled ? "bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed" : "bg-white border-gray-300 hover:border-blue-400"}`}
                >
                  {val}
                </button>
              );
            })}
          </div>
          <div className="text-sm text-gray-500 mb-6">{values.length}/3 selected</div>
          <button onClick={() => setStep(4)} disabled={values.length === 0} className="w-full py-3 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">Next</button>
        </div>
      )}

      {step === 4 && (
        <div>
          <h2 className="text-2xl font-bold mb-2">Life Situation</h2>
          <p className="text-gray-600 mb-6">Any constraints or context we should know about? (Optional)</p>
          <div className="mb-6 relative">
            <textarea
              value={lifeSituation}
              onChange={(e) => setLifeSituation(e.target.value.slice(0, 500))}
              placeholder="e.g. Married, 2 kids, living in Berlin, €40k savings, working as a software engineer"
              className="w-full h-32 p-3 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none text-black"
            />
            <div className="absolute bottom-3 right-3 text-sm text-gray-400">
              {lifeSituation.length}/500
            </div>
          </div>
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full py-3 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Saving..." : "Complete Setup"}
          </button>
        </div>
      )}
    </div>
  );
}
