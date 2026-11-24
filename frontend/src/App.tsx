import { BrowserRouter, Link, NavLink, Outlet, Route, Routes } from "react-router-dom"
import { AppStoreProvider } from "./store"
import { Dashboard } from "./pages/Dashboard"
import { NewResearch } from "./pages/NewResearch"
import { Reports } from "./pages/Reports"

function Layout() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" to="/">
            AgentFlow
          </Link>
          <nav className="nav" aria-label="Main">
            <NavLink to="/" end>
              Dashboard
            </NavLink>
            <NavLink to="/reports">Reports</NavLink>
          </nav>
          <Link className="btn btn-primary btn-sm" to="/new">
            New run
          </Link>
        </div>
      </header>
      <main id="main">
        <Outlet />
      </main>
      <footer className="footer">
        Local demo — no backend connected. Everything resets on reload.
      </footer>
    </>
  )
}

function NotFound() {
  return (
    <div className="page narrow">
      <p className="kicker">404</p>
      <h1>Page not found</h1>
      <p className="lede">That link does not point anywhere. Start from the dashboard.</p>
      <div className="actions mt-20">
        <Link className="btn btn-primary" to="/">
          Back to the dashboard
        </Link>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppStoreProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="new" element={<NewResearch />} />
            <Route path="reports" element={<Reports />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppStoreProvider>
  )
}
