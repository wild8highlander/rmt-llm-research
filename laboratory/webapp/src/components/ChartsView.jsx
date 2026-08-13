/**
 * ChartsView.jsx — Generated charts (PNG 600 DPI, PDF, SVG, Plotly HTML).
 */

import { BarChart3, FileImage, Download } from 'lucide-react'
import { useStore } from '../store'

const CHART_NAMES = [
  '01_loss_metrics', '02_eigenvalue_vs_mp', '03_confusion_matrix',
  '04_roc_deception', '05_hallucination_dist', '06_per_layer_gap',
  '07_reasoning_trace', '08_ncrit_threshold',
]

export default function ChartsView() {
  const { charts, results } = useStore()

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Generated Charts</h2>
        <p className="text-gray-400 text-sm">
          High-resolution charts from the latest run: PNG (600 DPI) + PDF + SVG + interactive Plotly HTML. Per-metric breakdown available.
        </p>
      </div>

      {!results ? (
        <div className="card flex items-center justify-center py-12">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 text-rmt-700 mx-auto mb-3" />
            <p className="text-gray-500">No charts yet. Run a scenario or experiment first.</p>
          </div>
        </div>
      ) : (
        <>
          {/* Chart format summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card">
              <div className="flex items-center gap-2 mb-2">
                <FileImage className="w-5 h-5 text-rmt-400" />
                <span className="text-sm text-gray-400">PNG (600 DPI)</span>
              </div>
              <div className="text-2xl font-bold text-rmt-300">{CHART_NAMES.length}</div>
              <div className="text-xs text-gray-500">raster</div>
            </div>
            <div className="card">
              <div className="flex items-center gap-2 mb-2">
                <FileImage className="w-5 h-5 text-green-400" />
                <span className="text-sm text-gray-400">PDF</span>
              </div>
              <div className="text-2xl font-bold text-green-300">{CHART_NAMES.length}</div>
              <div className="text-xs text-gray-500">vector</div>
            </div>
            <div className="card">
              <div className="flex items-center gap-2 mb-2">
                <FileImage className="w-5 h-5 text-purple-400" />
                <span className="text-sm text-gray-400">SVG</span>
              </div>
              <div className="text-2xl font-bold text-purple-300">{CHART_NAMES.length}</div>
              <div className="text-xs text-gray-500">vector</div>
            </div>
            <div className="card">
              <div className="flex items-center gap-2 mb-2">
                <FileImage className="w-5 h-5 text-orange-400" />
                <span className="text-sm text-gray-400">Plotly HTML</span>
              </div>
              <div className="text-2xl font-bold text-orange-300">1</div>
              <div className="text-xs text-gray-500">interactive</div>
            </div>
          </div>

          {/* Chart grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {CHART_NAMES.map((name) => (
              <div key={name} className="card">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-sm text-gray-200">{name.replace(/^\d+_/, '').replace(/_/g, ' ')}</h4>
                  <BarChart3 className="w-4 h-4 text-rmt-500" />
                </div>
                {/* Thumbnail placeholder — actual chart loaded from server */}
                <div className="aspect-video bg-rmt-950 rounded-lg flex items-center justify-center mb-3 border border-rmt-800">
                  <img
                    src={`/api/charts/${name}.png`}
                    alt={name}
                    className="max-w-full max-h-full"
                    onError={(e) => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'flex'
                    }}
                  />
                  <div className="hidden flex-col items-center text-gray-500 text-xs">
                    <FileImage className="w-8 h-8 mb-2" />
                    Generated after run
                  </div>
                </div>
                <div className="flex gap-2">
                  <a href={`/api/charts/${name}.png`} download className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1">
                    <Download className="w-3 h-3" /> PNG
                  </a>
                  <a href={`/api/charts/${name}.pdf`} download className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1">
                    <Download className="w-3 h-3" /> PDF
                  </a>
                  <a href={`/api/charts/${name}.svg`} download className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1">
                    <Download className="w-3 h-3" /> SVG
                  </a>
                </div>
              </div>
            ))}
          </div>

          {/* Interactive Plotly dashboard */}
          <div className="card">
            <h3 className="font-semibold text-lg mb-3">Interactive Dashboard (Plotly)</h3>
            <iframe
              src="/api/charts/interactive_dashboard.html"
              className="w-full h-[600px] bg-white rounded-lg"
              title="Interactive Dashboard"
            />
          </div>
        </>
      )}
    </div>
  )
}
