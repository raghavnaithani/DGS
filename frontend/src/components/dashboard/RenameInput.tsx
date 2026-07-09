"use client";

import { useState, useRef, useEffect } from "react";
import { fetchApi } from "@/lib/api";

interface RenameInputProps {
  sessionId: string;
  initialTitle: string;
  onSuccess: (newTitle: string) => void;
  onCancel: () => void;
}

export function RenameInput({ sessionId, initialTitle, onSuccess, onCancel }: RenameInputProps) {
  const [title, setTitle] = useState(initialTitle);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async () => {
    if (title.trim() === "" || title === initialTitle) {
      onCancel();
      return;
    }
    setIsSubmitting(true);
    try {
      await fetchApi(`/sessions/${sessionId}`, {
        method: "PATCH",
        body: JSON.stringify({ title: title.trim() }),
      });
      onSuccess(title.trim());
    } catch (error) {
      console.error("Failed to rename session", error);
      onCancel();
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    }
  };

  return (
    <input
      ref={inputRef}
      type="text"
      value={title}
      onChange={(e) => setTitle(e.target.value)}
      onBlur={handleSubmit}
      onKeyDown={handleKeyDown}
      disabled={isSubmitting}
      className="w-full bg-slate-800 text-slate-100 border border-indigo-500/50 rounded px-2 py-1 outline-none focus:ring-1 focus:ring-indigo-500 font-medium text-lg"
      onClick={(e) => e.stopPropagation()}
    />
  );
}
