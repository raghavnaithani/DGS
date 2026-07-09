"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { SessionCard, Session } from "@/components/dashboard/SessionCard";
import { UsageBanner } from "@/components/dashboard/UsageBanner";
import { fetchApi } from "@/lib/api";

interface DashboardClientProps {
  initialSessions: Session[];
  usage: {
    graphs_this_month: number;
    subscription_tier: string;
    free_limit: number;
  };
}

export function DashboardClient({ initialSessions, usage }: DashboardClientProps) {
  const [sessions, setSessions] = useState<Session[]>(initialSessions);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDomain, setSelectedDomain] = useState("All Domains");
  const [offset, setOffset] = useState(initialSessions.length);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(initialSessions.length === 20);

  const domains = useMemo(() => {
    const uniqueDomains = Array.from(new Set(sessions.map(s => s.domain)));
    return ["All Domains", ...uniqueDomains];
  }, [sessions]);

  const filteredSessions = useMemo(() => {
    return sessions.filter(session => {
      const matchesSearch = session.title.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesDomain = selectedDomain === "All Domains" || session.domain === selectedDomain;
      return matchesSearch && matchesDomain;
    });
  }, [sessions, searchQuery, selectedDomain]);

  const handleLoadMore = async () => {
    setIsLoadingMore(true);
    try {
      const data = await fetchApi(`/sessions?limit=20&offset=${offset}`);
      const newItems = data.items || [];
      if (newItems.length < 20) {
        setHasMore(false);
      }
      setSessions(prev => [...prev, ...newItems]);
      setOffset(prev => prev + newItems.length);
    } catch (error) {
      console.error("Failed to load more sessions", error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  const handleRename = (id: string, newTitle: string) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, title: newTitle } : s));
  };

  const handleDelete = (id: string) => {
    setSessions(prev => prev.filter(s => s.id !== id));
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
        <h1 className="text-3xl font-bold text-white">Your Decision Graphs</h1>
        <Link 
          href="/onboarding"
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-lg shadow-indigo-500/20"
        >
          New Graph
        </Link>
      </div>

      <UsageBanner 
        graphsThisMonth={usage.graphs_this_month}
        freeLimit={usage.free_limit || 5}
        subscriptionTier={usage.subscription_tier}
      />

      {sessions.length > 0 ? (
        <>
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-grow">
              <svg xmlns="http://www.w3.org/2000/svg" className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
              <input 
                type="text" 
                placeholder="Search graphs..." 
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            
            <select 
              value={selectedDomain}
              onChange={e => setSelectedDomain(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 min-w-[200px]"
            >
              {domains.map(domain => (
                <option key={domain} value={domain}>{domain}</option>
              ))}
            </select>
          </div>

          {filteredSessions.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredSessions.map(session => (
                <SessionCard 
                  key={session.id} 
                  session={session} 
                  onRename={handleRename}
                  onDelete={handleDelete}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-slate-900/50 rounded-xl border border-slate-800">
              <p className="text-slate-400">No graphs match your search.</p>
            </div>
          )}

          {hasMore && (
            <div className="mt-8 text-center">
              <button 
                onClick={handleLoadMore}
                disabled={isLoadingMore}
                className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                {isLoadingMore ? "Loading..." : "Load More"}
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-20 bg-slate-900/30 rounded-2xl border border-slate-800/50 border-dashed">
          <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
          </div>
          <h2 className="text-xl font-semibold text-white mb-2">No decision graphs yet</h2>
          <p className="text-slate-400 mb-6 max-w-sm mx-auto">
            Start a new simulation to generate your first decision tree and explore alternative futures.
          </p>
          <Link 
            href="/onboarding"
            className="inline-block bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors"
          >
            Start your first graph
          </Link>
        </div>
      )}
    </div>
  );
}
