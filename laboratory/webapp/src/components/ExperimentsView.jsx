/**
 * ExperimentsView.jsx — List and run research experiments.
 */

import { Play, Microscope, Clock, TrendingUp } from 'lucide-react'
import { useStore } from '../store'

export default function ExperimentsView() {
  const { experiments, startExperiment, isRunning, results } = useStore()

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Research Experiments</h2>
        <p className="text-gray-400 text-sm">
          Measurement-driven experiments for RMT-LLM theory verification. Each experiment produces spectral statistics, reasoning traces, and cross-implementation consistency checks.
        </p>
      </div>

      {experiments.length === 0 ? (
        <div className="card flex items-center justify-center py-12">
          <div className="text-center">
            <Microscope className="w-12 h-12 text-rmt-700 mx-auto mb-3" />
            <p className="text-gray-500">No experiments loaded.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {experiments.map((exp) => (
            <div key={exp.id} className="card hover:border-rmt-500 transition-colors">
              <div className="flex items-start gap-3 mb-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-rmt-600 to-accent-600">
                  <Microscope className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white">{exp.name}</h3>
                  <div className="text-xs text-gray-500 font-mono">{exp.id}</div>
                </div>
              </div>
              <p className="text-sm text-gray-400 mb-4">{exp.description}</p>
              <button
                onClick={() => startExperiment(exp.id)}
                disabled={isRunning}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                {isRunning ? 'Running...' : 'Run Experiment'}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Latest experiment results */}
      {results && results.elapsed_seconds !== undefined && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-rmt-400" />
            <h3 className="font-semibold text-lg">Last Run: {results.experiment_name}</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-rmt-950/50 rounded-lg p-3 flex items-center gap-3">
              <Clock className="w-5 h-5 text-rmt-400" />
              <div>
                <div className="text-xs text-gray-500">Elapsed</div>
                <div className="text-lg font-semibold text-rmt-300">{results.elapsed_seconds.toFixed(3)}s</div>
              </div>
            </div>
            {results.metrics && Object.entries(results.metrics).slice(0, 5).map(([k, v]) => (
              <div key={k} className="bg-rmt-950/50 rounded-lg p-3">
                <div className="text-xs text-gray-500 uppercase tracking-wide">{k.replace(/_/g, ' ')}</div>
                <div className="text-lg font-semibold text-rmt-300">
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
