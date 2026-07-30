// The backend is reached through THIS SAME origin — nginx (see
// frontend/nginx.conf) reverse-proxies /api/ and /ws/ to the backend
// container internally. This is what makes the app work identically on
// localhost:8009 and behind a real domain/reverse proxy: whatever port
// and protocol (http/https) the page itself loaded over is exactly what
// the API calls use too, so there's never a second port for an external
// proxy to have not terminated TLS for.
const API_BASE = "";
const WS_BASE = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
