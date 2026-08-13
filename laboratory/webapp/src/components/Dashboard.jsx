/**
 * Dashboard.jsx — Main dashboard with overview cards, quick stats, current experiment.
 */

import { useEffect } from 'react'
import {
  Activity, AlertTriangle, Brain, TrendingUp, Zap, Shield, Cpu,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts'
import { useStore } from '../store'

export default function Dashboard() {
  const { metricsStream, results, scenarios, experiments, isRunning, currentExperiment } = useStore()

  const stats = [
    {
      label: 'Scenarios Available',
      value: scenarios.length,
      icon: Brain,
      color: 'from-blue-500 to-cyan-500',
    },
    {
      label: 'Research Experiments',
      value: experiments.length,
      icon: Cpu,
      color: 'from-purple-500 to-pink-500',
    },
    {
      label: 'Live Metrics',
      value: metricsStream.length,
      icon: Activity,
      color: 'from-green-500 to-emerald-500',
    },
    {
      label: 'Avg Deception Score',
      value: results?.metrics?.mean_deception?.toFixed(3) || '—',
      icon: AlertTriangle,
      color: 'from-orange-500 to-red-500',
    },
  ]

  // Format metrics stream for chart
  const chartData = metricsStream.map((m, i) => ({
    step: i,
    honesty: m.honesty || 0,
    deception: m.deception || 0,
    hallucination: m.hallucination || 0,
  }))

  // Radar data from latest results
  const radarData = results ? [
    { metric: 'Honesty', value: results.metrics?.mean_honesty || 0, fullMark: 1 },
    { metric: 'Deception', value: results.metrics?.mean_deception || 0, fullMark: 1 },
    { metric: 'Hallucination', value: results.metrics?.mean_hallucination || 0, fullMark: 1 },
    { metric: 'Filter Bypass', value: Math.min(1, (results.metrics?.total_filter_bypasses || 0) / 10), fullMark: 1 },
    { metric: 'Match Rate', value: results.metrics?.match_rate || 0, fullMark: 1 },
  ] : []

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Research Dashboard</h2>
        <p className="text-gray-400 text-sm">
          Real-time monitoring of RMT-LLM experiments. Verifying: "models lie, hallucinate, store PII."
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => {
          const Icon = s.icon
          return (
            <div key={s.label} className="card relative overflow-hidden">
              <div className={`absolute top-0 right-0 w-20 h-20 rounded-full bg-gradient-to-br ${s.color} opacity-20 blur-2xl`} />
              <div className="flex items-start justify-between mb-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${s.color} bg-opacity-20`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
              </div>
              <div className="text-3xl font-bold text-white">{s.value}</div>
              <div className="text-sm text-gray-400 mt-1">{s.label}</div>
            </div>
          )
        })}
      </div>

      {/* Active experiment banner */}
      {isRunning && currentExperiment && (
        <div className="card border-yellow-700/50 bg-yellow-900/10">
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-yellow-400 animate-pulse" />
            <div>
              <div className="font-medium text-yellow-300">
                Running {currentExperiment.type}: {currentExperiment.id}
              </div>
              <div className="text-xs text-yellow-500/70">
                Streaming live metrics via Socket.io...
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live metrics chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-lg">Live Reasoning Trace</h3>
            <TrendingUp className="w-5 h-5 text-rmt-400" />
          </div>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="honestyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="deceptionGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="hallucGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a855f7" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5c" />
                <XAxis dataKey="step" stroke="#64748b" />
                <YAxis domain={[0, 1]} stroke="#64748b" />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5c', borderRadius: '8px' }}
                />
                <Area type="monotone" dataKey="honesty" stroke="#10b981" fill="url(#honestyGrad)" name="Honesty" />
                <Area type="monotone" dataKey="deception" stroke="#ef4444" fill="url(#deceptionGrad)" name="Deception" />
                <Area type="monotone" dataKey="hallucination" stroke="#a855f7" fill="url(#hallucGrad)" name="Hallucination" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
              No live metrics yet. Start a scenario or experiment.
            </div>
          )}
        </div>

        {/* Radar chart of latest results */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-lg">Latest Run Profile</h3>
            <Shield className="w-5 h-5 text-rmt-400" />
          </div>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1e3a5c" />
                <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
                <PolarRadiusAxis domain={[0, 1]} stroke="#64748b" />
                <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.4} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5c', borderRadius: '8px' }}
                />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
              No results yet. Run a scenario to populate.
            </div>
          )}
        </div>
      </div>

      {/* Latest results summary */}
      {results && (
        <div className="card">
          <h3 className="font-semibold text-lg mb-3">Latest Results: {results.scenario_name || results.experiment_name}</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {results.metrics && Object.entries(results.metrics).slice(0, 8).map(([k, v]) => (
              <div key={k} className="bg-rmt-950/50 rounded-lg p-3">
                <div className="text-xs text-gray-500 uppercase tracking-wide">{k.replace(/_/g, ' ')}</div>
                <div className="text-lg font-semibold text-rmt-300 mt-1">
                  {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
