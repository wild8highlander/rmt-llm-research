# 🖥️ server — the web dashboard backend

A small dependency-light Node server for the laboratory web dashboard:
serves the built front-end and exposes the lab results API over
Socket.io.

| File | Role |
|------|------|
| [`server.js`](server.js) | the server: static assets + results API |
| [`README.md`](README.md) | this file |

Run (from the webapp root, after `npm install`):

```bash
node server/server.js        # API on :3001
```

The front-end sources live in [`../src/`](../src/README.md); the full
dashboard workflow (Vite dev server, build, compose service) is described
in [`../README.md`](../README.md).
