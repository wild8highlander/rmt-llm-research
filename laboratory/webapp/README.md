# RMT-LLM Laboratory — Web App (CRA + Socket.io equivalent)

Real-time visualization dashboard for the RMT-LLM Laboratory. Built with **Vite + React 18 + Socket.io** (modern alternative to CRA + Socket.io, fully compatible).

## Features

- **Real-time experiment monitoring** via Socket.io (live metrics, logs, progress)
- **Interactive parameter system** — all bounds support `inf` for infinity
- **9 tabs**: Dashboard, Scenarios, Experiments, Parameters, Models, Live Monitor, Charts, Reports, Logs
- **Chart rendering**: Recharts (live) + Plotly (interactive HTML) + server-generated PNG/PDF/SVG at 600 DPI
- **13-format report browser** with one-click download
- **Model registry** — browse and download HuggingFace / ONNX models
- **Bilingual UI** — toggle between English and Russian

## Quick Start

```bash
# Install dependencies
npm install

# Start both Socket.io server (port 3001) and Vite dev server (port 5173)
npm start

# OR run them separately:
npm run server   # Socket.io server on :3001
npm run dev      # Vite dev server on :5173

# Production build
npm run build
npm run preview
```

Then open http://localhost:5173

## Architecture

```
webapp/
├── server/
│   └── server.js          # Express + Socket.io server, spawns Python lab
├── src/
│   ├── main.jsx           # React entry point
│   ├── App.jsx            # Root component with tab routing
│   ├── store.js           # Zustand global state + Socket.io client
│   ├── styles/index.css   # Tailwind CSS theme
│   └── components/
│       ├── Header.jsx     # Top bar with connection status
│       ├── Sidebar.jsx    # Tab navigation
│       ├── Dashboard.jsx  # Overview cards + live charts
│       ├── ScenariosView.jsx     # Pre-defined scenario runner
│       ├── ExperimentsView.jsx   # Research experiment runner
│       ├── ParametersView.jsx    # Infinite parameter editor
│       ├── ModelsView.jsx        # Model registry browser
│       ├── LiveMonitor.jsx       # Real-time metrics + reasoning trace
│       ├── ChartsView.jsx        # Generated chart gallery
│       ├── ReportsView.jsx       # 13-format report viewer
│       └── LogsView.jsx          # Full log viewer with search
├── package.json
├── vite.config.js
├── tailwind.config.js
└── index.html
```

## How it works

1. The browser loads the React SPA (Vite dev server, port 5173)
2. React connects to the Socket.io server (port 3001) via `socket.io-client`
3. User clicks "Run Scenario" → React emits `run_scenario` event
4. Server spawns the Python lab as a child process
5. Python streams logs/metrics back via stdout
6. Server forwards them as Socket.io events (`log`, `metric`, `progress`)
7. React updates the UI in real-time
8. On completion, server reads `results.json` and emits `results` event
9. React renders charts and reports

## Requirements

- Node.js 18+
- Python 3.10+ with the lab dependencies installed (`laboratory/python/lab_en/requirements.txt`)

## License

Proprietary — All rights reserved. Iskhak Hamzatovich Isaev, 2026.
