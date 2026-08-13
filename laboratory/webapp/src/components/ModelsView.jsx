/**
 * ModelsView.jsx — Model registry browser and downloader.
 */

import { Download, Cpu, HardDrive, ExternalLink, CheckCircle2 } from 'lucide-react'
import { useStore } from '../store'

const SOURCE_COLORS = {
  local: 'badge-success',
  huggingface: 'badge-warning',
  'onnx-model-zoo': 'badge-info',
  'keras-js': 'badge-info',
}

export default function ModelsView() {
  const { models, downloadModel } = useStore()

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Model Registry</h2>
        <p className="text-gray-400 text-sm">
          Browse and download small neural network weights from public registries. Hybrid approach: local TinyGPT for fast offline tests + HuggingFace / ONNX for richer experiments.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.length === 0 ? (
          <div className="card col-span-2 flex items-center justify-center py-12">
            <Cpu className="w-12 h-12 text-rmt-700 mr-3" />
            <p className="text-gray-500">No models loaded. Server may be offline.</p>
          </div>
        ) : (
          models.map((m) => {
            const sizeStr = m.params_count >= 1e6
              ? `${(m.params_count / 1e6).toFixed(1)}M`
              : m.params_count >= 1e3
                ? `${(m.params_count / 1e3).toFixed(0)}K`
                : String(m.params_count)
            const badge = SOURCE_COLORS[m.source] || 'badge-info'
            return (
              <div key={m.id} className="card hover:border-rmt-500 transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-lg text-white">{m.name}</h3>
                    <div className="text-xs text-gray-500 font-mono mt-0.5">{m.id}</div>
                  </div>
                  <span className={badge}>{m.source}</span>
                </div>

                <p className="text-sm text-gray-400 mb-3 line-clamp-2">{m.description}</p>

                <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
                  <div className="bg-rmt-950/50 rounded p-2">
                    <div className="text-gray-500">Params</div>
                    <div className="text-rmt-300 font-mono">{sizeStr}</div>
                  </div>
                  <div className="bg-rmt-950/50 rounded p-2">
                    <div className="text-gray-500">Format</div>
                    <div className="text-rmt-300 font-mono">{m.format}</div>
                  </div>
                  <div className="bg-rmt-950/50 rounded p-2">
                    <div className="text-gray-500">License</div>
                    <div className="text-rmt-300 font-mono text-[10px]">{m.license || 'unknown'}</div>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs text-gray-500 mb-3">
                  <HardDrive className="w-3 h-3" />
                  <span className="font-mono truncate">{m.url}</span>
                </div>

                <div className="flex items-center justify-between gap-2">
                  {m.source === 'local' ? (
                    <div className="badge-success flex items-center gap-1 flex-1 justify-center py-1.5">
                      <CheckCircle2 className="w-3 h-3" /> Available locally
                    </div>
                  ) : (
                    <button
                      onClick={() => downloadModel(m.id)}
                      className="btn-primary flex-1 flex items-center justify-center gap-2"
                    >
                      <Download className="w-4 h-4" /> Download
                    </button>
                  )}
                  {m.source !== 'local' && (
                    <a
                      href={m.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-secondary p-2"
                      title="Open source URL"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  )}
                </div>

                {m.languages_supported && (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {m.languages_supported.map((lang) => (
                      <span key={lang} className="text-[10px] px-1.5 py-0.5 rounded bg-rmt-800 text-gray-400 font-mono">
                        {lang}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
