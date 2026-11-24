import { Link } from "react-router-dom"
import { statusLabels } from "../data"
import { useStore } from "../store"

export function Dashboard() {
  const { runs, reports } = useStore()

  const open = runs.filter((run) => run.status !== "approved").length
  const words = runs.reduce((sum, run) => sum + run.words, 0)
  const sources = runs.reduce((sum, run) => sum + run.sources, 0)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="kicker">AgentFlow</p>
          <h1>Dashboard</h1>
          <p className="lede">
            Every run starts with a brief. A run moves from drafting to review,
            and its report is listed once it is signed.
          </p>
        </div>
        <Link className="btn btn-primary" to="/new">
          New run
        </Link>
      </div>

      <div className="stats">
        <div className="stat">
          <p className="value">{open}</p>
          <p className="label">Open runs</p>
        </div>
        <div className="stat">
          <p className="value">{words.toLocaleString()}</p>
          <p className="label">Words drafted</p>
        </div>
        <div className="stat">
          <p className="value">{sources}</p>
          <p className="label">Sources gathered</p>
        </div>
      </div>

      <section>
        <div className="section-head">
          <h2>All runs</h2>
          <span className="meta">{runs.length} total</span>
        </div>

        {runs.length === 0 ? (
          <p className="empty">No runs yet. Create one to get started.</p>
        ) : (
          <div className="list">
            {runs.map((run) => (
              <div className="list-row" key={run.id}>
                <span className="id">{run.id}</span>
                <span className="main">
                  <span className="title">{run.title}</span>
                  <span className="sub">
                    {run.format} · {run.length} ·{" "}
                    {run.words.toLocaleString()} words · {run.sources} sources ·{" "}
                    {run.createdAt}
                  </span>
                </span>
                <span className={`badge badge-${run.status}`}>
                  {statusLabels[run.status]}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      <p className="hint mt-20">
        {reports.length} signed {reports.length === 1 ? "report" : "reports"} in
        the archive. <Link to="/reports">View reports</Link>
      </p>
    </div>
  )
}
