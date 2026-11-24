/*
 * Types and demo content for the minimal app.
 * Replace this module with a real API when a backend is available.
 */

export type RunStatus = "drafting" | "review" | "approved"

export type Run = {
  id: string
  title: string
  audience: string
  format: string
  length: string
  status: RunStatus
  words: number
  sources: number
  createdAt: string
}

export type Report = {
  id: string
  title: string
  words: number
  sources: number
  signedAt: string
}

export type NewRunInput = {
  topic: string
  audience: string
  format: string
  length: string
}

export const statusLabels: Record<RunStatus, string> = {
  drafting: "Drafting",
  review: "Needs review",
  approved: "Approved",
}

export const formats = ["Technical article", "Research report", "Decision brief"]

export const lengths = ["Short", "Standard", "Long"]

export const seedRuns: Run[] = [
  {
    id: "0142",
    title: "Speculative decoding, tested against production traffic",
    audience: "Backend engineers running inference at scale",
    format: "Technical article",
    length: "Standard",
    status: "drafting",
    words: 1480,
    sources: 9,
    createdAt: "22 Nov 2026",
  },
  {
    id: "0141",
    title: "Vector index trade-offs at fifty million chunks",
    audience: "Search engineers",
    format: "Research report",
    length: "Standard",
    status: "drafting",
    words: 1640,
    sources: 12,
    createdAt: "21 Nov 2026",
  },
  {
    id: "0140",
    title: "EU AI Act: what shipping teams must document",
    audience: "Engineering leads shipping high-risk systems",
    format: "Decision brief",
    length: "Standard",
    status: "review",
    words: 2040,
    sources: 7,
    createdAt: "20 Nov 2026",
  },
]

export const seedReports: Report[] = [
  {
    id: "0139",
    title: "Batch inference economics: where the cost curve bends",
    words: 2340,
    sources: 8,
    signedAt: "20 Nov 2026",
  },
  {
    id: "0138",
    title: "Choosing an embedding model without a benchmark budget",
    words: 1910,
    sources: 11,
    signedAt: "18 Nov 2026",
  },
]
