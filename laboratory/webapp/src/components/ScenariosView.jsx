/**
 * ScenariosView.jsx — List and run pre-defined scenarios with synthetic NN.
 */

import { Play, AlertCircle, CheckCircle2, Brain } from 'lucide-react'
import { useStore } from '../store'

const BEHAVIOR_BADGES = {
  lie: { cls: 'badge-danger', label: 'Lie' },
  hallucinate: { cls: 'badge-warning', label: 'Hallucinate' },
  refuse: { cls: 'badge-info', label: 'Refuse' },
  truthful: { cls: 'badge-success', label: 'Truthful' },
  uncertain: { cls: 'badge-info', label: 'Uncertain' },
}

export default function ScenariosView() {
  const { scenarios, startScenario, isRunning, language } = useStore()

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Pre-defined Scenarios</h2>
        <p className="text-gray-400 text-sm">
          Run pre-built test cases against the synthetic TinyGPT neural network. Each scenario verifies a specific claim from the source news article.
        </p>
      </div>

      {scenarios.length === 0 ? (
        <div className="card flex items-center justify-center py-12">
          <div className="text-center">
            <Brain className="w-12 h-12 text-rmt-700 mx-auto mb-3" />
            <p className="text-gray-500">No scenarios loaded. Make sure the Socket.io server is running.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {scenarios.map((s) => {
            const badge = BEHAVIOR_BADGES[s.expected_behavior] || BEHAVIOR_BADGES.uncertain
            const desc = language === 'ru' ? (s.description_ru || s.description) : s.description
            const name = language === 'ru' ? (s.name_ru || s.name) : s.name
            return (
              <div key={s.id} className="card hover:border-rmt-500 transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="text-xs text-gray-500 font-mono mb-1">{s.id}</div>
                    <h3 className="font-semibold text-lg text-white">{name}</h3>
                  </div>
                  <span className={badge.cls}>{badge.label}</span>
                </div>
                <p className="text-sm text-gray-400 mb-4 line-clamp-3">{desc}</p>
                <div className="text-xs text-gray-500 mb-4">
                  <span className="font-mono">{s.prompts?.length || 0} prompts</span>
                  {s.parameter_overrides && (
                    <span> · overrides: {Object.keys(s.parameter_overrides).length}</span>
                  )}
                </div>
                <button
                  onClick={() => startScenario(s.id)}
                  disabled={isRunning}
                  className="btn-primary w-full flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  {isRunning ? 'Running...' : 'Run Scenario'}
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* What each scenario verifies */}
      <div className="card bg-rmt-950/40">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rmt-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-medium text-rmt-300 mb-2">What these scenarios verify</h4>
            <ul className="text-sm text-gray-400 space-y-1.5">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5" />
                Models already know the answer but generate plausible wrong reasoning (SCEN-LIE-01)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5" />
                Long-form generation past N_crit produces fabricated facts (SCEN-HALL-02)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5" />
                Models internally plan to deceive the user (SCEN-DECEIT-03)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5" />
                Memorized PII / API keys leak from model weights (SCEN-DATA-04)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5" />
                Safety filters fire only at output time, not at reasoning (SCEN-FILTER-05)
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
