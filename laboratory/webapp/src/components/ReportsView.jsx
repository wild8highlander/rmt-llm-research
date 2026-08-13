/**
 * ReportsView.jsx — 13-format report viewer and downloader.
 */

import { FileText, Download, FileCode, Database, FileSpreadsheet } from 'lucide-react'
import { useStore } from '../store'

const REPORT_FORMATS = [
  { ext: 'txt', name: 'Plain Text', icon: FileText, color: 'text-gray-400', desc: 'Simple text, no formatting' },
  { ext: 'md', name: 'Markdown', icon: FileText, color: 'text-blue-400', desc: 'GitHub-flavored markdown' },
  { ext: 'csv', name: 'CSV', icon: FileSpreadsheet, color: 'text-green-400', desc: 'Comma-separated values' },
  { ext: 'html', name: 'HTML', icon: FileCode, color: 'text-orange-400', desc: 'Self-contained HTML page' },
  { ext: 'json', name: 'JSON', icon: FileCode, color: 'text-yellow-400', desc: 'Structured JSON data' },
  { ext: 'pdf', name: 'PDF', icon: FileText, color: 'text-red-400', desc: 'Portable Document Format' },
  { ext: 'docx', name: 'Word DOCX', icon: FileText, color: 'text-blue-500', desc: 'Microsoft Word document' },
  { ext: 'yaml', name: 'YAML', icon: FileCode, color: 'text-purple-400', desc: 'YAML Ain\'t Markup Language' },
  { ext: 'xml', name: 'XML', icon: FileCode, color: 'text-green-500', desc: 'eXtensible Markup Language' },
  { ext: 'tex', name: 'LaTeX', icon: FileCode, color: 'text-pink-400', desc: 'Academic publishing format' },
  { ext: 'parquet', name: 'Parquet', icon: Database, color: 'text-cyan-400', desc: 'Columnar big-data format' },
  { ext: 'xlsx', name: 'Excel', icon: FileSpreadsheet, color: 'text-emerald-400', desc: 'Microsoft Excel workbook' },
  { ext: 'sqlite', name: 'SQLite', icon: Database, color: 'text-indigo-400', desc: 'SQL-queryable database' },
]

export default function ReportsView() {
  const { reports, results } = useStore()

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-gradient mb-1">Reports (13 formats)</h2>
        <p className="text-gray-400 text-sm">
          Every run generates reports in 13 formats. Each report contains: (1) detailed results with explanations, (2) full task launch logs.
        </p>
      </div>

      {!results ? (
        <div className="card flex items-center justify-center py-12">
          <div className="text-center">
            <FileText className="w-12 h-12 text-rmt-700 mx-auto mb-3" />
            <p className="text-gray-500">No reports yet. Run a scenario or experiment first.</p>
          </div>
        </div>
      ) : (
        <>
          {/* Format summary */}
          <div className="card">
            <h3 className="font-semibold text-lg mb-3">All 13 report formats generated</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {REPORT_FORMATS.map((f) => {
                const Icon = f.icon
                return (
                  <div key={f.ext} className="bg-rmt-950/50 rounded-lg p-3 hover:bg-rmt-800/50 transition-colors">
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className={`w-4 h-4 ${f.color}`} />
                      <span className="text-xs font-mono text-gray-400">.{f.ext}</span>
                    </div>
                    <div className="text-xs font-medium text-gray-200">{f.name}</div>
                    <div className="text-[10px] text-gray-500 mt-1">{f.desc}</div>
                    {reports[f.ext] && (
                      <a
                        href={`/api/reports/${reports[f.ext].split('/').pop()}`}
                        download
                        className="mt-2 btn-secondary text-xs w-full flex items-center justify-center gap-1"
                      >
                        <Download className="w-3 h-3" /> Download
                      </a>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Latest results preview */}
          <div className="card">
            <h3 className="font-semibold text-lg mb-3">Latest Results Preview</h3>
            <div className="bg-rmt-950 rounded-lg p-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <div className="text-xs text-gray-500 uppercase">Experiment</div>
                  <div className="text-sm text-rmt-300">{results.experiment_name || '—'}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 uppercase">Language</div>
                  <div className="text-sm text-rmt-300">{results.language || '—'} / {results.version || '—'}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 uppercase">Scenario</div>
                  <div className="text-sm text-rmt-300">{results.scenario_name || '—'}</div>
                </div>
              </div>

              {results.metrics && (
                <div className="mt-4">
                  <div className="text-xs text-gray-500 uppercase mb-2">Metrics</div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {Object.entries(results.metrics).slice(0, 8).map(([k, v]) => (
                      <div key={k} className="bg-rmt-900 rounded p-2">
                        <div className="text-[10px] text-gray-500">{k.replace(/_/g, ' ')}</div>
                        <div className="text-sm font-mono text-rmt-300">
                          {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Report structure explanation */}
          <div className="card bg-rmt-950/40">
            <h3 className="font-semibold text-lg mb-3">Report Structure</h3>
            <div className="space-y-3 text-sm text-gray-400">
              <div className="border-l-2 border-rmt-500 pl-3">
                <div className="font-medium text-rmt-300">Part I — Detailed Results with Explanations</div>
                <ul className="mt-1 text-xs space-y-1">
                  <li>• Experiment metadata (name, language, version, scenario)</li>
                  <li>• Key metrics table</li>
                  <li>• RMT spectral analysis (MP bounds, eigenvalues, signal detection)</li>
                  <li>• Reasoning trace summary (honesty, deception, hallucination)</li>
                  <li>• Interpretation prose explaining what the numbers mean</li>
                </ul>
              </div>
              <div className="border-l-2 border-accent-500 pl-3">
                <div className="font-medium text-accent-400">Part II — Full Task Launch Logs</div>
                <ul className="mt-1 text-xs space-y-1">
                  <li>• Timestamped log lines from the entire run</li>
                  <li>• Scenario loading, parameter merging, model instantiation</li>
                  <li>• Per-prompt generation results and timing</li>
                  <li>• Spectral analysis per layer</li>
                  <li>• Error messages and warnings (if any)</li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
