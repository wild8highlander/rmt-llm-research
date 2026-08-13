/**
 * Header.jsx — Top header with connection status, language toggle, branding.
 */

import { Activity, Globe, Wifi, WifiOff } from 'lucide-react'
import { useStore } from '../store'

export default function Header() {
  const { connected, language, setLanguage, isRunning } = useStore()

  return (
    <header className="bg-rmt-950/80 backdrop-blur-md border-b border-rmt-800 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rmt-500 to-accent-500 flex items-center justify-center shadow-lg">
          <Activity className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-gradient">
            RMT-LLM Laboratory
          </h1>
          <p className="text-xs text-gray-400">
            Random Matrix Theory × Large Language Models
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {isRunning && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-900/30 border border-yellow-700/50">
            <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
            <span className="text-xs text-yellow-300 font-medium">Experiment running</span>
          </div>
        )}

        <button
          onClick={() => setLanguage(language === 'en' ? 'ru' : 'en')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rmt-800 hover:bg-rmt-700 border border-rmt-700 transition-colors"
        >
          <Globe className="w-4 h-4" />
          <span className="text-sm font-medium uppercase">{language}</span>
        </button>

        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
          connected ? 'bg-green-900/30 border border-green-700/50' : 'bg-red-900/30 border border-red-700/50'
        }`}>
          {connected ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-400" />
          )}
          <span className={`text-xs font-medium ${connected ? 'text-green-300' : 'text-red-300'}`}>
            {connected ? 'Connected' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  )
}
