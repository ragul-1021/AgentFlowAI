import { useState } from "react"
import type { FormEvent } from "react"
import { Link, useNavigate } from "react-router-dom"
import { formats, lengths } from "../data"
import { useStore } from "../store"

function wordTarget(length: string) {
  if (length === "Short") return 800
  if (length === "Long") return 3400
  return 2100
}

export function NewResearch() {
  const { createRun } = useStore()
  const navigate = useNavigate()

  const [topic, setTopic] = useState("")
  const [audience, setAudience] = useState("")
  const [format, setFormat] = useState(formats[0])
  const [length, setLength] = useState("Standard")
  const [error, setError] = useState("")

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!topic.trim()) {
      setError("Describe what the run should research.")
      return
    }
    if (!audience.trim()) {
      setError("Name the reader.")
      return
    }

    createRun({
      topic: topic.trim(),
      audience: audience.trim(),
      format,
      length,
    })
    navigate("/")
  }

  return (
    <div className="page narrow">
      <div className="page-head">
        <div>
          <p className="kicker">New run</p>
          <h1>What should it research?</h1>
          <p className="lede">
            Give the brief and the reader. Everything else is derived from them.
          </p>
        </div>
      </div>

      <form className="card" onSubmit={handleSubmit} noValidate>
        <div className="field">
          <label htmlFor="topic">Subject</label>
          <textarea
            id="topic"
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
            placeholder="What is the piece about, and what should a reader be able to do after finishing it?"
          />
        </div>

        <div className="field">
          <label htmlFor="audience">Reader</label>
          <input
            id="audience"
            className="input"
            value={audience}
            onChange={(event) => setAudience(event.target.value)}
            placeholder="Backend engineers running inference at scale"
          />
        </div>

        <div className="two">
          <div className="field">
            <label htmlFor="format">Format</label>
            <select
              id="format"
              className="select"
              value={format}
              onChange={(event) => setFormat(event.target.value)}
            >
              {formats.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="length">Length</label>
            <select
              id="length"
              className="select"
              value={length}
              onChange={(event) => setLength(event.target.value)}
            >
              {lengths.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
          </div>
        </div>

        <p className="hint">
          Estimate · about {wordTarget(length).toLocaleString()} words
        </p>

        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}

        <div className="actions">
          <button className="btn btn-primary" type="submit">
            Create run
          </button>
          <Link className="btn btn-quiet" to="/">
            Cancel
          </Link>
        </div>
      </form>
    </div>
  )
}
