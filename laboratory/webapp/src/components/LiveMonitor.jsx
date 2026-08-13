/**
 * LiveMonitor.jsx — Real-time metrics stream and reasoning trace.
 */

import { Radio, Activity, Square } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { useStore } from '../store'

export default function LiveMonitor() {
  const { metricsStream, logs, isRunning, stopExperiment, currentExperiment, progress } = useStore()

  const chartData = metricsStream.map((m, i) => ({
    step: i,
    honesty: m.honesty || 0,
    deception: m.deception || 0,
    hallucination: m.hallucination || 0,
    temperature: m.temperature || 0.7,
  }))

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gradient mb-1 flex items-center gap-2">
            <Radio className="w-6 h-6" /> Live Monitor
          </h2>
          <p className="text-gray-400 text-sm">
            Real-time metrics streamed via Socket.io. Watch the model's hidden reasoning evolve step-by-step.
          </p>
        </div>
        {isRunning && (
          <button onClick={stopExperiment} className="btn-danger flex items-center gap-2">
            <Square className="w-4 h-4" /> Stop
          </button>
        )}
      </div>

      {isRunning && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-yellow-400 animate-pulse" />
              <span className="text-sm text-yellow-300">
                Running: {currentExperiment?.type} / {currentExperiment?.id}
              </span>
            </div>
            <span className="text-sm font-mono text-rmt-300">{progress.toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-rmt-950 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-rmt-500 to-accent-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time reasoning trace chart */}
        <div className="card">
          <h3 className="font-semibold text-lg mb-4">Hidden Reasoning Trace (real-time)</h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5c" />
                <XAxis dataKey="step" stroke="#64748b" />
                <YAxis domain={[0, 1]} stroke="#64748b" />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5c', borderRadius: '8px' }}
                />
                <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'threshold', fill: '#ef4444', fontSize: 10 }} />
                <Line type="monotone" dataKey="honesty" stroke="#10b981" strokeWidth={2} dot={false} name="Honesty" />
                <Line type="monotone" dataKey="deception" stroke="#ef4444" strokeWidth={2} dot={false} name="Deception" />
                <Line type="monotone" dataKey="hallucination" stroke="#a855f7" strokeWidth={2} dot={false} name="Hallucination" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-500 text-sm">
              No live data. Start an experiment from the Scenarios or Experiments tab.
            </div>
          )}
        </div>

        {/* Latest reasoning thoughts */}
        <div className="card">
          <h3 className="font-semibold text-lg mb-4">Latest "Thoughts"</h3>
          <div className="space-y-2 max-h-[300px] overflow-auto">
            {metricsStream.length > 0 ? (
              metricsStream.slice(-10).reverse().map((m, i) => (
                <div key={i} className="bg-rmt-950/50 rounded-lg p-3 border-l-2 border-rmt-700">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-gray-500">step {metricsStream.length - i}</span>
                    <div className="flex gap-2">
                      <span className={`badge text-[10px] ${m.deception > 0.5 ? 'badge-danger' : 'badge-success'}`}>
                        D={(m.deception || 0).toFixed(2)}
                      </span>
                      <span className={`badge text-[10px] ${m.hallucination > 0.5 ? 'badge-warning' : 'badge-info'}`}>
                        H={(m.hallucination || 0).toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-300">{m.thought || '(no thought captured)'}</p>
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-sm text-center py-8">
                No thoughts captured yet.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Live log stream */}
      <div className="card">
        <h3 className="font-semibold text-lg mb-3">Live Log Stream</h3>
        <div className="bg-rmt-950 rounded-lg p-4 max-h-64 overflow-auto font-mono text-xs">
          {logs.length > 0 ? (
            logs.slice(-50).map((line, i) => (
              <div key={i} className="text-gray-300 hover:bg-rmt-900 px-1 py-0.5 rounded">
                <span className="text-rmt-500">›</span> {line}
              </div>
            ))
          ) : (
            <div className="text-gray-500">No logs yet.</div>
          )}
        </div>
      </div>
    </div>
  )
}
