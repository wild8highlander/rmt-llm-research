# 🧩 src — the web dashboard front-end

React sources of the laboratory web app: the visualization surfaces that
render spectral experiments and live results.

| Path | Role |
|------|------|
| [`App.jsx`](App.jsx) | root component / route layout |
| [`main.jsx`](main.jsx) | entry point |
| [`store.js`](store.js) | state store (actions + socket wiring) |
| [`components/`](components/) | UI components |
| [`styles/`](styles/) | stylesheets |
| [`README.md`](README.md) | this file |

Develop and build (from the webapp root):

```bash
npm run dev      # Vite dev server on :5173
npm run build    # production bundle
```

The backend that feeds it lives in [`../server/`](../server/README.md);
see [`../README.md`](../README.md) for the full dashboard guide.
