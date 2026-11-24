# AgentFlow — frontend

A minimal React 19 + TypeScript + Vite app: a dashboard, a new-run form, and a
reports list.

## Commands

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # tsc -b && vite build
npm run lint     # oxlint
npm run preview
```

## Routes

| Route | Screen |
| --- | --- |
| `/` | Dashboard: stats and every run |
| `/new` | Form to create a run |
| `/reports` | Signed reports |

## Source layout

```
src/
  main.tsx          React entry, loads the stylesheet
  App.tsx           router, top bar, footer, 404
  data.ts           types and demo content
  store.tsx         runs/reports state and createRun
  pages/            Dashboard, NewResearch, Reports
  styles/           small plain-CSS stylesheet
```

## State

The app is a local preview. New runs are kept in memory and reset on reload;
no backend is connected.
