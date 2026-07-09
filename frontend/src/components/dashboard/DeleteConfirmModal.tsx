"use client";

import { useState } from "react";
import { fetchApi } from "@/lib/api";

interface DeleteConfirmModalProps {
  sessionId: string;
  title: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteConfirmModal({ sessionId, title, onConfirm, onCancel }: DeleteConfirmModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await fetchApi(`/sessions/${sessionId}`, {
        method: "DELETE",
      });
      onConfirm();
    } catch (error) {
      console.error("Failed to delete session", error);
      setIsDeleting(false);
      onCancel();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={onCancel}>
      <div 
        className="bg-slate-900 border border-slate-700 rounded-xl p-6 max-w-md w-full shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Delete this graph?</h2>
        <p className="text-slate-400 mb-6">
          Are you sure you want to delete <span className="text-slate-200 font-medium">"{title}"</span>? This action cannot be undone.
        </p>
        
        <div className="flex justify-end gap-3">
          <button 
            onClick={onCancel}
            disabled={isDeleting}
            className="px-4 py-2 text-slate-300 hover:text-white font-medium transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={handleDelete}
            disabled={isDeleting}
            className="px-4 py-2 bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
