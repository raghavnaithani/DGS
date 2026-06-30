"use client";

import { useEffect, useMemo, useState } from "react";

import { Background, Controls, ReactFlow, type Edge, type Node, useEdgesState, useNodesState } from "@xyflow/react";

import { type GraphData, type GraphNode } from "@/lib/api";
import { DecisionNodeCard } from "./DecisionNodeCard";
import { NodeDetailPanel } from "./NodeDetailPanel";

type GraphCanvasProps = {
  graph: GraphData;
  readOnly?: boolean;
  onBranch?: (parentNodeId: string, actionDescription: string) => Promise<void>;
};

type DecisionNodeData = {
  node: GraphNode;
  onSelect: (nodeId: string) => void;
};

const nodeTypes = {
  decisionNode: DecisionNodeCard,
};

function toFlowNodes(graph: GraphData, onSelect: (nodeId: string) => void): Node<DecisionNodeData>[] {
  const parentByNodeId = new Map<string, string>();
  for (const edge of graph.edges) {
    if (!parentByNodeId.has(edge.target)) {
      parentByNodeId.set(edge.target, edge.source);
    }
  }

  const depthByNodeId = new Map<string, number>();
  const resolveDepth = (nodeId: string): number => {
    const cachedDepth = depthByNodeId.get(nodeId);
    if (cachedDepth !== undefined) {
      return cachedDepth;
    }

    const parentId = parentByNodeId.get(nodeId);
    if (!parentId) {
      depthByNodeId.set(nodeId, 0);
      return 0;
    }

    const depth = resolveDepth(parentId) + 1;
    depthByNodeId.set(nodeId, depth);
    return depth;
  };

  const nodesByDepth = new Map<number, GraphNode[]>();
  for (const node of graph.nodes) {
    const depth = resolveDepth(node.id);
    const existing = nodesByDepth.get(depth) ?? [];
    existing.push(node);
    nodesByDepth.set(depth, existing);
  }

  const positions = new Map<string, { x: number; y: number }>();
  const xGap = 360;
  const yGap = 200;
  for (const [depth, nodesAtDepth] of Array.from(nodesByDepth.entries()).sort(([left], [right]) => left - right)) {
    const total = nodesAtDepth.length;
    nodesAtDepth.forEach((node, index) => {
      positions.set(node.id, {
        x: depth * xGap,
        y: (index - (total - 1) / 2) * yGap,
      });
    });
  }

  return graph.nodes.map((node, index) => ({
    id: node.id,
    type: "decisionNode",
    position: positions.get(node.id) ?? {
      x: (node.time_step ?? index) * 340,
      y: (index % 3) * 190,
    },
    data: {
      node,
      onSelect,
    },
  }));
}

function toFlowEdges(graph: GraphData): Edge[] {
  return graph.edges.map((edge, index) => ({
    id: edge.id || `${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    label: edge.action_description,
    style: {
      stroke: "#94a3b8",
      strokeWidth: 1.5,
    },
    labelStyle: {
      fill: "#475569",
      fontSize: 11,
    },
    animated: false,
  }));
}

export function GraphCanvas({ graph, readOnly = false, onBranch }: GraphCanvasProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(graph.nodes[0]?.id ?? null);
  const [branchActionDescription, setBranchActionDescription] = useState("");
  const [branching, setBranching] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<DecisionNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    const handleSelect = (nodeId: string) => {
      setSelectedNodeId(nodeId);
    };

    const nextNodes = toFlowNodes(graph, handleSelect);
    const nextEdges = toFlowEdges(graph);
    setNodes(nextNodes);
    setEdges(nextEdges);
    setSelectedNodeId((current) => (current && nextNodes.some((node) => node.id === current) ? current : nextNodes[0]?.id ?? null));
  }, [graph, setEdges, setNodes]);

  const selectedNode = useMemo(() => graph.nodes.find((node) => node.id === selectedNodeId) ?? null, [graph.nodes, selectedNodeId]);

  const handleBranch = async () => {
    if (readOnly || !onBranch || !selectedNodeId) {
      return;
    }

    if (!branchActionDescription.trim()) {
      return;
    }

    setBranching(true);
    try {
      await onBranch(selectedNodeId, branchActionDescription.trim());
      setBranchActionDescription("");
    } finally {
      setBranching(false);
    }
  };

  if (!graph.nodes.length) {
    return <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">No graph data returned.</div>;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-h-[70vh] rounded-xl border border-slate-200 bg-white">
        <div className="h-[70vh] min-h-[480px]">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            fitView
            minZoom={0.2}
            maxZoom={2}
            className="bg-slate-50"
          >
            <Background gap={24} size={1} color="#e2e8f0" />
            <Controls position="bottom-right" />
          </ReactFlow>
        </div>
      </div>

      <NodeDetailPanel
        node={selectedNode}
        readOnly={readOnly}
        actionDescription={branchActionDescription}
        onActionDescriptionChange={setBranchActionDescription}
        onBranch={handleBranch}
        branching={branching}
      />
    </div>
  );
}
