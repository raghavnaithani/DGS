"use client";

import { useState } from "react";
import Link from "next/link";
import { RenameInput } from "./RenameInput";
import { DeleteConfirmModal } from "./DeleteConfirmModal";

export interface Session {
  id: string;
  title: string;
  domain: string;
  horizon_months: number;
  node_count: number;
  created_at: string;
}

interface SessionCardProps {
  session: Session;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
}

const DOMAIN_COLORS = [
  "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
  "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  "bg-rose-500/10 text-rose-400 border-rose-500/20",
  "bg-amber-500/10 text-amber-400 border-amber-500/20",
  "bg-violet-500/10 text-violet-400 border-violet-500/20",
  "bg-slate-500/10 text-slate-400 border-slate-500/20",
];

function getDomainColor(domain: string) {
  let hash = 0;
  for (let i = 0; i < domain.length; i++) {
    hash = domain.charCodeAt(i) + ((hash << 5) - hash);
  }
  return DOMAIN_COLORS[Math.abs(hash) % DOMAIN_COLORS.length];
}

function getRelativeTime(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) return "just now";
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  return `${Math.floor(diffInSeconds / 2592000)} months ago`;
}

export function SessionCard({ session, onRename, onDelete }: SessionCardProps) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  return (
    <>
      <div className="group relative flex flex-col bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-5 transition-all">
        <div className="flex items-start justify-between mb-3">
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${getDomainColor(session.domain)} capitalize`}>
            {session.domain}
          </span>
          
          <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
            <Link 
              href={`/graph/${session.id}`}
              className="p-1.5 text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/10 rounded transition-colors"
              title="Open Graph"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </Link>
            <button 
              onClick={(e) => { e.preventDefault(); setIsRenaming(true); }}
              className="p-1.5 text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 rounded transition-colors"
              title="Rename"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/></svg>
            </button>
            <button 
              onClick={(e) => { e.preventDefault(); setIsDeleting(true); }}
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
              title="Delete"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
            </button>
          </div>
        </div>

        {isRenaming ? (
          <RenameInput 
            sessionId={session.id}
            initialTitle={session.title}
            onSuccess={(newTitle) => {
              onRename(session.id, newTitle);
              setIsRenaming(false);
            }}
            onCancel={() => setIsRenaming(false)}
          />
        ) : (
          <Link href={`/graph/${session.id}`} className="block focus:outline-none flex-grow">
            <h3 className="text-lg font-semibold text-slate-100 truncate mb-1" title={session.title}>
              {session.title}
            </h3>
          </Link>
        )}

        <div className="text-sm text-slate-500 mt-4 flex items-center gap-2">
          <span>{session.node_count} nodes</span>
          <span className="w-1 h-1 bg-slate-700 rounded-full"></span>
          <span>{session.horizon_months} months</span>
          <span className="w-1 h-1 bg-slate-700 rounded-full"></span>
          <span>{getRelativeTime(session.created_at)}</span>
        </div>
      </div>

      {isDeleting && (
        <DeleteConfirmModal 
          sessionId={session.id}
          title={session.title}
          onConfirm={() => {
            onDelete(session.id);
            setIsDeleting(false);
          }}
          onCancel={() => setIsDeleting(false)}
        />
      )}
    </>
  );
}
