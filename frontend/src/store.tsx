import { createContext, useCallback, useContext, useMemo, useState } from "react"
import type { ReactNode } from "react"
import type { NewRunInput, Report, Run } from "./data"
import { seedReports, seedRuns } from "./data"

type Store = {
  runs: Run[]
  reports: Report[]
  createRun: (input: NewRunInput) => Run
}

const StoreContext = createContext<Store | null>(null)

function today() {
  return new Date().toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  })
}

export function AppStoreProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<Run[]>(seedRuns)
  const [reports] = useState<Report[]>(seedReports)

  const createRun = useCallback(
    (input: NewRunInput) => {
      const highest = runs.reduce(
        (max, run) => Math.max(max, Number(run.id)),
        140,
      )
      const run: Run = {
        id: String(highest + 1).padStart(4, "0"),
        title: input.topic,
        audience: input.audience,
        format: input.format,
        length: input.length,
        status: "drafting",
        words: 0,
        sources: 0,
        createdAt: today(),
      }

      setRuns((current) => [run, ...current])
      return run
    },
    [runs],
  )

  const value = useMemo<Store>(
    () => ({ runs, reports, createRun }),
    [runs, reports, createRun],
  )

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>
}

export function useStore() {
  const store = useContext(StoreContext)
  if (!store) {
    throw new Error("useStore must be used inside AppStoreProvider")
  }
  return store
}
