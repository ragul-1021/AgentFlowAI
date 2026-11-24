import { Link } from "react-router-dom"
import { useStore } from "../store"

export function Reports() {
  const { reports } = useStore()

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="kicker">Archive</p>
          <h1>Reports</h1>
          <p className="lede">Signed reports from finished runs.</p>
        </div>
        <Link className="btn" to="/new">
          New run
        </Link>
      </div>

      {reports.length === 0 ? (
        <p className="empty">Nothing signed yet.</p>
      ) : (
        <div className="list">
          {reports.map((report) => (
            <div className="list-row" key={report.id}>
              <span className="id">{report.id}</span>
              <span className="main">
                <span className="title">{report.title}</span>
                <span className="sub">
                  {report.words.toLocaleString()} words · {report.sources}{" "}
                  sources · signed {report.signedAt}
                </span>
              </span>
              <span className="badge badge-approved">Signed</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
