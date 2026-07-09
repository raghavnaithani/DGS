import { DashboardClient } from "./DashboardClient";
import { fetchServerApi } from "@/lib/server-api";

export const metadata = {
  title: "Dashboard - Decision Graph Simulator",
};

export default async function DashboardPage() {
  // Fetch sessions and usage server-side
  let initialSessions = [];
  let usage = { graphs_this_month: 0, subscription_tier: "free", free_limit: 5 };

  try {
    const [sessionsData, usageData] = await Promise.all([
      fetchServerApi("/sessions?limit=20&offset=0"),
      fetchServerApi("/account/usage")
    ]);

    initialSessions = sessionsData.items || [];
    usage = {
      graphs_this_month: usageData.graphs_this_month || 0,
      subscription_tier: usageData.subscription_tier || "free",
      free_limit: usageData.free_limit || 5,
    };
  } catch (error) {
    console.error("Failed to fetch dashboard data:", error);
  }

  return <DashboardClient initialSessions={initialSessions} usage={usage} />;
}
