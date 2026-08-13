/**
 * server.js — Socket.io server for RMT-LLM Laboratory webapp.
 *
 * Provides:
 *   - REST API: /api/scenarios, /api/experiments, /api/models, /api/parameters
 *   - Static file serving: /api/charts/*, /api/reports/*
 *   - Socket.io events: run_scenario, run_experiment, stop_experiment,
 *     download_model, save_config
 *   - Real-time streaming: log, metric, progress, results, reports, charts
 *
 * Spawns the Python lab as a child process for actual computation.
 */

import express from 'express'
import http from 'http'
import cors from 'cors'
import { Server } from 'socket.io'
import { spawn, execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const LAB_ROOT = path.resolve(__dirname, '..', '..')
const RESULTS_DIR = path.join(LAB_ROOT, 'results')
const CHARTS_DIR = path.join(RESULTS_DIR, 'charts')
const REPORTS_DIR = path.join(RESULTS_DIR, 'reports')
const SHARED_DIR = path.join(LAB_ROOT, 'shared')

// Python lab path (use EN by default)
const PYTHON_LAB = path.join(LAB_ROOT, 'python', 'lab_en')

const PORT = process.env.PORT || 3001

const app = express()
app.use(cors())
app.use(express.json())

const server = http.createServer(app)
const io = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
})

// ---------------------------------------------------------------------------
// REST API
// ---------------------------------------------------------------------------

app.get('/api/health', (req, res) => {
  res.json({ ok: true, ts: new Date().toISOString() })
})

app.get('/api/scenarios', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(SHARED_DIR, 'scenarios.json'), 'utf8'))
    res.json(data)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/experiments', (req, res) => {
  // Return the 5 experiments defined in research.py
  res.json({
    experiments: [
      { id: '1', name: 'Spectral Signature', description: 'Compute spectral signature of TinyGPT hidden activations.' },
      { id: '2', name: 'N_crit Sweep', description: 'Sweep token counts and detect RMT-predicted hallucination onset.' },
      { id: '3', name: 'Deception Detection', description: 'Probe TinyGPT for deceptive reasoning patterns.' },
      { id: '4', name: 'PII Leakage', description: 'Probe TinyGPT weights for memorized PII.' },
      { id: '5', name: 'Cross-Implementation Verification', description: 'Verify MP bounds empirically vs theoretically.' },
    ],
  })
})

app.get('/api/models', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(SHARED_DIR, 'model_registry.json'), 'utf8'))
    res.json(data)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/parameters', (req, res) => {
  // Return the default parameter space (mirror of Python parameters.py)
  res.json({
    defaults: {
      temperature: 0.7, max_tokens: 256, top_k: 50, top_p: 0.95,
      context_window: 1024, ncrit_threshold: 114.0, theta_b_deg: 7.07,
      beta_caputo: 0.5, rlhf_pressure: 0.0, n_layers: 6, hidden_dim: 64,
      n_heads: 4, vocab_size: 256, seed: 42, epochs: 3,
      learning_rate: 0.001, batch_size: 4,
      enable_filter: true, capture_hidden: true, language: 'en',
    },
  })
})

// Static file serving for charts and reports
app.use('/api/charts', express.static(CHARTS_DIR))
app.use('/api/reports', express.static(REPORTS_DIR))

// ---------------------------------------------------------------------------
// Socket.io events
// ---------------------------------------------------------------------------

let runningChild = null

io.on('connection', (socket) => {
  console.log(`[socket.io] client connected: ${socket.id}`)
  socket.emit('log', `[SERVER] client connected: ${socket.id}`)

  socket.on('run_scenario', (data) => {
    console.log('[run_scenario]', data)
    runPython(socket, 'scenario', data)
  })

  socket.on('run_experiment', (data) => {
    console.log('[run_experiment]', data)
    runPython(socket, 'experiment', data)
  })

  socket.on('stop_experiment', () => {
    if (runningChild) {
      runningChild.kill('SIGTERM')
      socket.emit('log', '[SERVER] experiment stopped by user')
      runningChild = null
    }
  })

  socket.on('download_model', (data) => {
    console.log('[download_model]', data)
    socket.emit('log', `[SERVER] model download requested: ${data.modelId}`)
    // Spawn Python to handle the download
    const child = spawn('python', ['-c', `
import sys; sys.path.insert(0, '${PYTHON_LAB}')
import model_downloader as md
rep = md.fetch_model('${data.modelId}', dest_dir='${path.join(RESULTS_DIR, 'models')}')
print(repr(rep))
`])
    child.stdout.on('data', (d) => socket.emit('log', `[PYTHON] ${d.toString().trim()}`))
    child.stderr.on('data', (d) => socket.emit('log', `[PYTHON-ERR] ${d.toString().trim()}`))
    child.on('close', (code) => {
      socket.emit('log', `[SERVER] download finished (exit ${code})`)
    })
  })

  socket.on('save_config', (params) => {
    const cfgPath = path.join(RESULTS_DIR, 'logs', `config_${Date.now()}.json`)
    fs.writeFileSync(cfgPath, JSON.stringify(params, null, 2))
    socket.emit('log', `[SERVER] config saved: ${cfgPath}`)
  })

  socket.on('disconnect', () => {
    console.log(`[socket.io] client disconnected: ${socket.id}`)
  })
})

// ---------------------------------------------------------------------------
// Run Python lab as child process
// ---------------------------------------------------------------------------

function runPython(socket, kind, data) {
  if (runningChild) {
    socket.emit('log', '[SERVER] another experiment is already running')
    return
  }

  // Write a temporary config file
  const cfgPath = path.join(RESULTS_DIR, 'logs', `run_${Date.now()}.json`)
  const cfg = {
    parameters: data.parameters || {},
    [kind + '_id']: data.scenarioId || data.experimentId,
  }
  fs.writeFileSync(cfgPath, JSON.stringify(cfg, null, 2))

  // Build Python command that streams output
  const pythonScript = `
import sys, json, os, datetime
sys.path.insert(0, '${PYTHON_LAB}')
os.chdir('${PYTHON_LAB}')
import parameters as P
import charts
import reports
import research
import scenarios as scen

cfg = json.load(open('${cfgPath}'))
params = cfg.get('parameters', {})
space = {p.name: p for p in P.default_parameter_space()}
validated = {}
for k, v in params.items():
    if k in space:
        try: validated[k] = space[k].validate(v)
        except: validated[k] = space[k].default
    else: validated[k] = v

logs = []
def emit_log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    logs.append(line)
    print(line, flush=True)

emit_log(f'Running ${kind}: ${data.scenarioId || data.experimentId}')

try:
    if '${kind}' == 'scenario':
        sc = scen.get_scenario('${data.scenarioId}')
        if sc is None: raise ValueError('Scenario not found')
        results = scen.run_scenario(sc, validated, logs=logs)
    else:
        results = research.run_experiment('${data.experimentId}', validated)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{ts}_{'${data.scenarioId || data.experimentId}'}"
    res_path = '${REPORTS_DIR}/' + name + '_results.json'
    with open(res_path, 'w') as f: json.dump(results, f, indent=2, default=str)
    emit_log(f'Results saved: {res_path}')

    charts_dir = '${CHARTS_DIR}/' + name
    os.makedirs(charts_dir, exist_ok=True)
    written_charts = charts.generate_all_charts(results, charts_dir)
    emit_log(f'Charts: {sum(len(v) for v in written_charts.values())} files')

    written_reports = reports.generate_all_reports(results, logs,
        out_dir='${REPORTS_DIR}', experiment_name=name)
    emit_log(f'Reports: {len(written_reports)} formats')

    print('RESULTS_JSON:' + res_path, flush=True)
except Exception as e:
    emit_log(f'[ERROR] {type(e).__name__}: {e}')
    import traceback; traceback.print_exc()
`

  const child = spawn('python', ['-u', '-c', pythonScript], {
    cwd: PYTHON_LAB,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  })
  runningChild = child

  let progress = 0
  let resultsPath = null

  child.stdout.on('data', (d) => {
    const text = d.toString()
    text.split('\n').forEach((line) => {
      line = line.trim()
      if (!line) return
      if (line.startsWith('RESULTS_JSON:')) {
        resultsPath = line.substring('RESULTS_JSON:'.length)
      } else {
        socket.emit('log', line)
        // Mock progress based on log lines
        progress = Math.min(95, progress + Math.random() * 5)
        socket.emit('progress', progress)
        // Mock metric stream
        if (line.includes('[SCENARIO]') || line.includes('generated')) {
          socket.emit('metric', {
            honesty: Math.random() * 0.6,
            deception: 0.3 + Math.random() * 0.5,
            hallucination: Math.random() * 0.7,
            temperature: 0.7,
            thought: 'Streaming metric from server...',
          })
        }
      }
    })
  })

  child.stderr.on('data', (d) => {
    socket.emit('log', `[PYTHON-ERR] ${d.toString().trim()}`)
  })

  child.on('close', (code) => {
    runningChild = null
    socket.emit('log', `[SERVER] process exited (code ${code})`)
    socket.emit('progress', 100)

    // Load results and emit
    if (resultsPath && fs.existsSync(resultsPath)) {
      try {
        const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'))
        socket.emit('results', results)

        // Find the latest charts and reports
        const chartDir = path.dirname(resultsPath).replace('reports', 'charts')
        const baseName = path.basename(resultsPath).replace('_results.json', '')
        const chartSubdir = path.join(chartDir, baseName)
        const reportFiles = {}
        if (fs.existsSync(chartSubdir)) {
          const files = fs.readdirSync(chartSubdir)
          const charts = { png: [], pdf: [], svg: [], html: [] }
          files.forEach((f) => {
            const ext = f.split('.').pop()
            if (charts[ext]) charts[ext].push(path.join(chartSubdir, f).replace(LAB_ROOT, ''))
          })
          socket.emit('charts', charts)
        }
        // List report files matching the base name
        if (fs.existsSync(REPORTS_DIR)) {
          const allReports = fs.readdirSync(REPORTS_DIR)
          allReports.forEach((f) => {
            if (f.startsWith(baseName + '.')) {
              const ext = f.split('.').pop()
              reportFiles[ext] = path.join(REPORTS_DIR, f)
            }
          })
          socket.emit('reports', reportFiles)
        }
      } catch (e) {
        socket.emit('log', `[SERVER] failed to load results: ${e.message}`)
      }
    }
  })
}

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

server.listen(PORT, () => {
  console.log(`\n========================================================`)
  console.log(`  RMT-LLM Laboratory Socket.io server`)
  console.log(`  Listening on http://localhost:${PORT}`)
  console.log(`  Lab root: ${LAB_ROOT}`)
  console.log(`  Python lab: ${PYTHON_LAB}`)
  console.log(`========================================================\n`)
})

process.on('SIGINT', () => {
  console.log('\n[server] shutting down...')
  if (runningChild) runningChild.kill('SIGTERM')
  process.exit(0)
})
