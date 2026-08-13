/**
 * store.js — Zustand global state for the RMT-LLM Laboratory webapp.
 * Manages: connection status, active experiments, parameters, results, logs.
 */

import { create } from 'zustand'
import { io } from 'socket.io-client'

const socket = io({
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 1000,
})

export const useStore = create((set, get) => ({
  // ----- Connection -----
  socket,
  connected: false,
  setConnected: (c) => set({ connected: c }),

  // ----- Active view / tab -----
  activeTab: 'dashboard',
  setActiveTab: (t) => set({ activeTab: t }),

  // ----- Language -----
  language: 'en',
  setLanguage: (l) => set({ language: l }),

  // ----- Parameters -----
  parameters: {},
  setParameter: (key, value) => set((s) => ({
    parameters: { ...s.parameters, [key]: value },
  })),
  setParameters: (params) => set({ parameters: params }),
  resetParameters: () => set({ parameters: {} }),

  // ----- Scenarios -----
  scenarios: [],
  setScenarios: (s) => set({ scenarios: s }),

  // ----- Experiments -----
  experiments: [],
  setExperiments: (e) => set({ experiments: e }),

  // ----- Models (registry) -----
  models: [],
  setModels: (m) => set({ models: m }),

  // ----- Live experiment state -----
  isRunning: false,
  setIsRunning: (r) => set({ isRunning: r }),
  currentExperiment: null,
  setCurrentExperiment: (e) => set({ currentExperiment: e }),
  progress: 0,
  setProgress: (p) => set({ progress: p }),

  // ----- Real-time metrics stream -----
  metricsStream: [],
  addMetric: (m) => set((s) => ({
    metricsStream: [...s.metricsStream.slice(-99), m],
  })),
  clearMetrics: () => set({ metricsStream: [] }),

  // ----- Results -----
  results: null,
  setResults: (r) => set({ results: r }),

  // ----- Logs -----
  logs: [],
  addLog: (line) => set((s) => ({
    logs: [...s.logs.slice(-499), line],
  })),
  clearLogs: () => set({ logs: [] }),

  // ----- Reports -----
  reports: {},
  setReports: (r) => set({ reports: r }),

  // ----- Charts -----
  charts: {},
  setCharts: (c) => set({ charts: c }),

  // ----- Actions -----
  emit: (event, data) => {
    const s = get().socket
    if (s.connected) s.emit(event, data)
  },

  startScenario: (scenarioId) => {
    const params = get().parameters
    get().clearMetrics()
    get().clearLogs()
    get().setIsRunning(true)
    get().setCurrentExperiment({ type: 'scenario', id: scenarioId })
    get().emit('run_scenario', { scenarioId, parameters: params })
  },

  startExperiment: (experimentId) => {
    const params = get().parameters
    get().clearMetrics()
    get().clearLogs()
    get().setIsRunning(true)
    get().setCurrentExperiment({ type: 'experiment', id: experimentId })
    get().emit('run_experiment', { experimentId, parameters: params })
  },

  stopExperiment: () => {
    get().emit('stop_experiment', {})
    get().setIsRunning(false)
  },

  downloadModel: (modelId) => {
    get().emit('download_model', { modelId })
  },

  fetchScenarios: async () => {
    try {
      const res = await fetch('/api/scenarios')
      const data = await res.json()
      set({ scenarios: data.scenarios || [] })
    } catch (e) { console.error('fetchScenarios:', e) }
  },

  fetchExperiments: async () => {
    try {
      const res = await fetch('/api/experiments')
      const data = await res.json()
      set({ experiments: data.experiments || [] })
    } catch (e) { console.error('fetchExperiments:', e) }
  },

  fetchModels: async () => {
    try {
      const res = await fetch('/api/models')
      const data = await res.json()
      set({ models: data.models || [] })
    } catch (e) { console.error('fetchModels:', e) }
  },

  fetchParameterSpace: async () => {
    try {
      const res = await fetch('/api/parameters')
      const data = await res.json()
      set({ parameters: data.defaults || {} })
    } catch (e) { console.error('fetchParameterSpace:', e) }
  },
}))

// Socket event listeners
socket.on('connect', () => useStore.getState().setConnected(true))
socket.on('disconnect', () => useStore.getState().setConnected(false))
socket.on('log', (line) => useStore.getState().addLog(line))
socket.on('metric', (m) => useStore.getState().addMetric(m))
socket.on('progress', (p) => useStore.getState().setProgress(p))
socket.on('results', (r) => {
  useStore.getState().setResults(r)
  useStore.getState().setIsRunning(false)
  useStore.getState().setProgress(100)
})
socket.on('reports', (r) => useStore.getState().setReports(r))
socket.on('charts', (c) => useStore.getState().setCharts(c))
socket.on('error', (err) => {
  console.error('Server error:', err)
  useStore.getState().setIsRunning(false)
})
