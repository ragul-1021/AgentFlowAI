// Citation ↔ evidence linking for the report workspace prototype.
(function () {
  const notes = new Map(
    Array.from(document.querySelectorAll(".ev-note")).map((n) => [n.dataset.key, n])
  );
  if (!notes.size) return;

  let current = null;

  function activate(key) {
    const note = notes.get(key);
    if (!note || note === current) return;

    if (current) current.classList.remove("is-active");
    note.classList.add("is-active");
    current = note;

    document.querySelectorAll(".cite.is-active").forEach((c) => c.classList.remove("is-active"));
    const cite = document.querySelector('.cite[data-cite="' + key + '"]');
    if (cite) cite.classList.add("is-active");
  }

  document.querySelectorAll("[data-cite]").forEach((cite) => {
    const key = cite.dataset.cite;
    cite.addEventListener("click", (event) => {
      event.preventDefault();
      activate(key);
      notes.get(key)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    cite.addEventListener("mouseenter", () => activate(key));
    cite.addEventListener("focus", () => activate(key));
  });

  const hash = window.location.hash.replace("#", "");
  if (hash) {
    activate(hash);
  } else {
    const first = document.querySelector(".cite[data-cite]");
    if (first) activate(first.dataset.cite);
  }
})();
