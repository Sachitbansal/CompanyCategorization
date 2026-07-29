// Backend is always reached on the same hostname the page was loaded
// from, just on port 8006 — this holds whether you're on localhost or a
// real server, without hardcoding an IP/domain anywhere.
const API_BASE = `${location.protocol}//${location.hostname}:8006`;
const WS_BASE = `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:8006`;
