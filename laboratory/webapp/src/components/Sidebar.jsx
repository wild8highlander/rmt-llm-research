/**
 * Sidebar.jsx — Tab navigation with icons.
 */

import {
  LayoutDashboard, FlaskConical, Microscope, Sliders,
  Download, Radio, BarChart3, FileText, ScrollText,
} from 'lucide-react'
import { useStore } from '../store'

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'scenarios', label: 'Scenarios', icon: FlaskConical },
  { id: 'experiments', label: 'Experiments', icon: Microscope },
  { id: 'parameters', label: 'Parameters', icon: Sliders },
  { id: 'models', label: 'Models', icon: Download },
  { id: 'live', label: 'Live Monitor', icon: Radio },
  { id: 'charts', label: 'Charts', icon: BarChart3 },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'logs', label: 'Logs', icon: ScrollText },
]

export default function Sidebar() {
  const { activeTab, setActiveTab } = useStore()

  return (
    <aside className="w-64 bg-rmt-950/50 border-r border-rmt-800 flex flex-col">
      <nav className="flex-1 p-3 space-y-1">
        {TABS.map((tab) => {
          const Icon = tab.icon
          const active = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all ${
                active
                  ? 'bg-rmt-700/50 text-rmt-200 shadow-md'
                  : 'text-gray-400 hover:bg-rmt-800/50 hover:text-gray-200'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="text-sm font-medium">{tab.label}</span>
            </button>
          )
        })}
      </nav>

      <div className="p-3 border-t border-rmt-800">
        <div className="text-xs text-gray-500 space-y-1">
          <div className="font-medium text-gray-400">Author</div>
          <div>Iskhak Hamzatovich Isaev</div>
          <div>ORCID: 0009-0003-7299-0701</div>
          <div className="pt-2 text-[10px]">v1.0.0 · Proprietary License</div>
        </div>
      </div>
    </aside>
  )
}
