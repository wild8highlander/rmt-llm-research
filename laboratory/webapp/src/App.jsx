/**
 * App.jsx — Root component for RMT-LLM Laboratory webapp.
 * Renders the main dashboard with tab navigation.
 */

import { useEffect } from 'react'
import { useStore } from './store'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Dashboard from './components/Dashboard'
import ScenariosView from './components/ScenariosView'
import ExperimentsView from './components/ExperimentsView'
import ParametersView from './components/ParametersView'
import ModelsView from './components/ModelsView'
import ChartsView from './components/ChartsView'
import ReportsView from './components/ReportsView'
import LogsView from './components/LogsView'
import LiveMonitor from './components/LiveMonitor'

export default function App() {
  const { activeTab, fetchScenarios, fetchExperiments, fetchModels, fetchParameterSpace } = useStore()

  useEffect(() => {
    fetchScenarios()
    fetchExperiments()
    fetchModels()
    fetchParameterSpace()
  }, [fetchScenarios, fetchExperiments, fetchModels, fetchParameterSpace])

  const renderTab = () => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard />
      case 'scenarios': return <ScenariosView />
      case 'experiments': return <ExperimentsView />
      case 'parameters': return <ParametersView />
      case 'models': return <ModelsView />
      case 'live': return <LiveMonitor />
      case 'charts': return <ChartsView />
      case 'reports': return <ReportsView />
      case 'logs': return <LogsView />
      default: return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          {renderTab()}
        </main>
      </div>
    </div>
  )
}
