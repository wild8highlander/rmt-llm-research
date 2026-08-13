/**
 * LogsView.jsx — Full log viewer with search and filter.
 */

import { useState } from 'react'
import { ScrollText, Search, Trash2, Download } from 'lucide-react'
import { useStore } from '../store'

export default function LogsView() {
  const { logs, clearLogs } = useStore()
  const [filter, setFilter] = useState('')
  const [level, setLevel] = useState('all')

  const filtered = logs.filter((line) => {
    if (level !== 'all') {
      if (level === 'error' && !line.includes('[ERROR]') && !line.includes('[ОШИБКА]')) return false
      if (level === 'warn' && !line.includes('[WARN]')) return false
      if (level === 'scenario' && !line.includes('[SCENARIO]')) return false
    }
    if (filter && !line.toLowerCase().includes(filter.toLowerCase())) return false
    return true
  })

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gradient mb-1 flex items-center gap-2">
            <ScrollText className="w-6 h-6" /> Task Launch Logs
          </h2>
          <p className="text-gray-400 text-sm">
            Full timestamped logs from every experiment and scenario run.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => {
              const blob = new Blob([logs.join('\n')], { type: 'text/plain' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `rmt_llm_logs_${Date.now()}.log`
              a.click()
            }}
            className="btn-secondary flex items-center gap-2"
          >
            <Download className="w-4 h-4" /> Export
          </button>
          <button onClick={clearLogs} className="btn-danger flex items-center gap-2">
            <Trash2 className="w-4 h-4" /> Clear
          </button>
        </div>
      </div>

      <div className="card">
        <div className="flex gap-3 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search logs..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="input w-full pl-10"
            />
          </div>
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            className="input"
          >
            <option value="all">All levels</option>
            <option value="scenario">Scenario</option>
            <option value="warn">Warnings</option>
            <option value="error">Errors</option>
          </select>
        </div>

        <div className="bg-rmt-950 rounded-lg p-4 max-h-[600px] overflow-auto font-mono text-xs">
          {filtered.length > 0 ? (
            filtered.map((line, i) => {
              let cls = 'text-gray-300'
              if (line.includes('[ERROR]') || line.includes('[ОШИБКА]')) cls = 'text-red-400'
              else if (line.includes('[WARN]')) cls = 'text-yellow-400'
              else if (line.includes('[SCENARIO]') || line.includes('[СЦЕНАРИЙ]')) cls = 'text-blue-400'
              else if (line.includes('match_rate') || line.includes('completed')) cls = 'text-green-400'
              return (
                <div key={i} className={`${cls} hover:bg-rmt-900 px-1 py-0.5 rounded`}>
                  {line}
                </div>
              )
            })
          ) : (
            <div className="text-gray-500 text-center py-8">No logs match the filter.</div>
          )}
        </div>

        <div className="mt-3 text-xs text-gray-500 flex justify-between">
          <span>{filtered.length} of {logs.length} lines</span>
          <span>Auto-scrolls to bottom on new lines</span>
        </div>
      </div>
    </div>
  )
}
