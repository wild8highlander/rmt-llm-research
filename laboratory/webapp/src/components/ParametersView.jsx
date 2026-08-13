/**
 * ParametersView.jsx — Infinite parameter system. All bounds support inf.
 */

import { useState } from 'react'
import { Sliders, Infinity as InfinityIcon, RotateCcw, Save } from 'lucide-react'
import { useStore } from '../store'

const PARAM_METADATA = [
  { name: 'temperature', type: 'float', min: 0, max: Infinity, default: 0.7, step: 0.01, desc: 'Sampling temperature (0 = greedy, inf = pure random)' },
  { name: 'max_tokens', type: 'int', min: 1, max: Infinity, default: 256, step: 1, desc: 'Maximum tokens to generate' },
  { name: 'top_k', type: 'int', min: 0, max: Infinity, default: 50, step: 1, desc: 'Top-k filtering (0 = disabled, inf = no filter)' },
  { name: 'top_p', type: 'float', min: 0, max: 1.0, default: 0.95, step: 0.01, desc: 'Nucleus sampling probability mass' },
  { name: 'context_window', type: 'int', min: 1, max: Infinity, default: 1024, step: 1, desc: 'Context window size (tokens)' },
  { name: 'ncrit_threshold', type: 'float', min: 0, max: Infinity, default: 114.0, step: 0.1, desc: 'RMT critical token count threshold' },
  { name: 'theta_b_deg', type: 'float', min: 0, max: 360, default: 7.07, step: 0.01, desc: 'BBP rotation angle (degrees)' },
  { name: 'beta_caputo', type: 'float', min: 0, max: Infinity, default: 0.5, step: 0.01, desc: 'Caputo fractional memory parameter' },
  { name: 'rlhf_pressure', type: 'float', min: 0, max: Infinity, default: 0.0, step: 0.01, desc: 'RLHF drift strength — accelerates hallucination' },
  { name: 'n_layers', type: 'int', min: 1, max: Infinity, default: 6, step: 1, desc: 'Number of transformer layers' },
  { name: 'hidden_dim', type: 'int', min: 1, max: Infinity, default: 64, step: 1, desc: 'Hidden dimension of synthetic model' },
  { name: 'n_heads', type: 'int', min: 1, max: Infinity, default: 4, step: 1, desc: 'Number of attention heads' },
  { name: 'vocab_size', type: 'int', min: 1, max: Infinity, default: 256, step: 1, desc: 'Vocabulary size of synthetic model' },
  { name: 'seed', type: 'int', min: 0, max: Infinity, default: 42, step: 1, desc: 'Random seed' },
  { name: 'epochs', type: 'int', min: 0, max: Infinity, default: 3, step: 1, desc: 'Training epochs for synthetic model' },
  { name: 'learning_rate', type: 'float', min: 0, max: Infinity, default: 0.001, step: 0.0001, desc: 'Learning rate' },
  { name: 'batch_size', type: 'int', min: 1, max: Infinity, default: 4, step: 1, desc: 'Batch size' },
  { name: 'enable_filter', type: 'bool', default: true, desc: 'Enable output safety filter' },
  { name: 'capture_hidden', type: 'bool', default: true, desc: 'Capture hidden reasoning trace' },
  { name: 'language', type: 'categorical', default: 'en', choices: ['en', 'ru'], desc: 'Output language' },
]

export default function ParametersView() {
  const { parameters, setParameter, resetParameters, emit } = useStore()
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    emit('save_config', parameters)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gradient mb-1">Infinite Parameter System</h2>
          <p className="text-gray-400 text-sm">
            All numeric parameters support <code className="text-rmt-300">inf</code> for unbounded values. Range [0, ∞) by default.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={resetParameters} className="btn-secondary flex items-center gap-2">
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
          <button onClick={handleSave} className="btn-primary flex items-center gap-2">
            <Save className="w-4 h-4" /> {saved ? 'Saved!' : 'Save Config'}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center gap-2 mb-4 text-rmt-300">
          <InfinityIcon className="w-5 h-5" />
          <span className="font-medium">All bounds accept the string "inf" for infinity</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PARAM_METADATA.map((p) => {
            const value = parameters[p.name] !== undefined ? parameters[p.name] : p.default
            return (
              <div key={p.name} className="bg-rmt-950/50 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <label className="text-sm font-medium text-gray-200">{p.name}</label>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="badge-info">{p.type}</span>
                      {p.type !== 'bool' && p.type !== 'categorical' && (
                        <span className="text-xs text-gray-500 font-mono">
                          [{p.min === 0 ? '0' : p.min}, {p.max === Infinity ? '∞' : p.max}]
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mb-3">{p.desc}</p>

                {p.type === 'bool' ? (
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={Boolean(value)}
                      onChange={(e) => setParameter(p.name, e.target.checked)}
                      className="w-4 h-4 rounded accent-rmt-500"
                    />
                    <span className="text-sm text-gray-300">{value ? 'true' : 'false'}</span>
                  </label>
                ) : p.type === 'categorical' ? (
                  <select
                    value={String(value)}
                    onChange={(e) => setParameter(p.name, e.target.value)}
                    className="input w-full"
                  >
                    {p.choices.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    value={value === Infinity ? 'inf' : value}
                    step={p.step}
                    min={p.min === 0 ? 0 : undefined}
                    onChange={(e) => {
                      const raw = e.target.value
                      if (raw.toLowerCase() === 'inf' || raw === '∞') {
                        setParameter(p.name, Infinity)
                      } else {
                        const num = p.type === 'int' ? parseInt(raw, 10) : parseFloat(raw)
                        if (!isNaN(num)) setParameter(p.name, num)
                      }
                    }}
                    className="input w-full"
                  />
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="card bg-rmt-950/40">
        <div className="flex items-start gap-3">
          <Sliders className="w-5 h-5 text-rmt-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-gray-400">
            <p className="mb-2">
              <strong className="text-rmt-300">Infinite parameter semantics:</strong>
            </p>
            <ul className="space-y-1 list-disc list-inside">
              <li>Numeric parameters accept any value ≥ 0, including <code className="text-rmt-300">inf</code> for true infinity</li>
              <li><code className="text-rmt-300">temperature = 0</code> means greedy decoding; <code className="text-rmt-300">temperature = inf</code> means pure random sampling</li>
              <li><code className="text-rmt-300">top_k = 0</code> disables top-k filtering; <code className="text-rmt-300">top_k = inf</code> disables any filtering</li>
              <li><code className="text-rmt-300">rlhf_pressure = inf</code> drives Caputo mean collapse time <code className="text-rmt-300">⟨T_crit⟩ → 0</code></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
