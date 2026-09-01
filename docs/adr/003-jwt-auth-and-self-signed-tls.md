# ADR 003: JWT auth and self-signed TLS for Card 3.3

**Context:**
Card 3.3 requires authentication, input validation and rate limiting before exposing
`/predict` and `/models` outside the compose network, plus TLS termination at the
nginx proxy. Two open questions needed a team decision: static API key vs JWT, and
self-signed vs a real certificate.

**Decision:**
- **Auth: JWT (HS256)**, not a static API key. Closer to a production auth flow
  (expiry via `exp` claim, per-subject tokens) at negligible extra complexity over
  a shared header secret. The signing secret (`JWT_SECRET_KEY`) is a single shared
  team secret for now — per-member keys are out of scope for this card.
- **Rate-limit store: in-memory**, scoped per client IP, reset on process restart.
  Sufficient at course scale (single API replica). Documented as a caveat in the
  README; a redis-backed store is deferred to Card 3.4 once replicas > 1.
- **TLS cert: self-signed**, generated locally and mounted into the nginx `proxy`
  service via `deploy/nginx/certs/` (git-ignored). A real cert (Let's Encrypt via
  cert-manager) is deferred to the k8s ingress in Card 3.4.

**Consequences:**
- No external dependency (redis, cert authority) needed to land this card.
- The in-memory rate limiter does **not** survive a restart and does **not**
  coordinate across multiple API replicas — must be swapped for redis before
  Card 3.4 scales `api` beyond 1 replica.
- The rate limiter keys on `X-Real-IP` (nginx's `$remote_addr`), not
  `X-Forwarded-For` — the latter's first hop is client-suppliable and would let 
  a caller spoof a fresh bucket on every request.
- Browsers/clients hitting the compose stack directly will see a certificate
  warning until the k8s ingress brings a real cert; acceptable for the course demo.