const app = document.querySelector("#app");
const state = { timer: null, liveHistory: new Map(), liveMatch: null };

const esc = (value) => String(value ?? "—")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const pct = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;
const relativeAge = (value) => {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60000));
  if (minutes < 1) return "à l’instant";
  if (minutes < 60) return `il y a ${minutes} min`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `il y a ${hours} h` : `il y a ${Math.round(hours / 24)} j`;
};
const api = async (path, options) => {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || "Erreur API");
  }
  return response.json();
};
const apiMaybe = async (path) => {
  const response = await fetch(`/api${path}`);
  if (response.status === 404) return null;
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || "Erreur API");
  }
  return response.json();
};
const surfaceColor = (surface) => surface === "Clay" ? "#dc6b42" : surface === "Grass" ? "#a8cf4d" : "#6ba8ad";
const routeId = (id) => encodeURIComponent(id);

function tournamentCard(tournament) {
  return `<a class="card tournament-card" style="--surface-color:${surfaceColor(tournament.surface)}" href="#/tournament/${routeId(tournament.id)}">
    <span class="badge">${esc(tournament.surface)} · ${esc(tournament.start_date.slice(0,4))}</span>
    <h3>${esc(tournament.name)}</h3>
    <div class="meta"><span>${tournament.players} joueurs</span><span>${tournament.matches} matchs</span></div>
    <div class="card-footer"><span>${tournament.replay_matches} replays live</span><span class="round-chip">Ouvrir →</span></div>
  </a>`;
}

async function renderHome() {
  const data = await api("/home");
  const replay = data.replay;
  app.innerHTML = `<div class="page">
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Tennis intelligence / modèles figés</div>
        <h1>Chaque point<br><span class="accent">change le match.</span></h1>
        <p>Explorez les tableaux, les probabilités pré-match et les mouvements live à travers un replay historique entièrement reproductible.</p>
        <div class="actions"><a class="btn dark" href="#/tournament/${routeId(replay.tournament_id)}">Reprendre le replay</a><a class="btn dark" href="#/tournaments">Explorer les tournois</a></div>
      </div>
      <div class="court-card" aria-label="Court de tennis stylisé">
        <div class="court-lines"></div><div class="orb"></div>
        <div class="stat"><strong>4</strong><span>moteurs officiels connectés</span></div>
      </div>
    </section>
    <section class="replay-banner">
      <div><div class="eyebrow">Replay actif · ${esc(replay.next_round || "Terminé")}</div><h2>${esc(replay.tournament.name)}</h2><p>${esc(replay.tournament.surface)} · ${esc(replay.tournament.start_date)} · Tournament Adapter désactivé</p></div>
      <div class="actions"><a class="btn" href="#/tournament/${routeId(replay.tournament_id)}">Voir le tableau</a></div>
    </section>
    <div class="section-head"><div><div class="eyebrow">Historical Replay</div><h2>Tournois à explorer</h2></div><a class="text-link" href="#/tournaments">Tout voir</a></div>
    <div class="grid grid-3">${data.tournaments.slice(0,6).map(tournamentCard).join("")}</div>
    <div class="section-head"><div><div class="eyebrow">Point-by-point disponible</div><h2>Matchs à rejouer</h2></div></div>
    <div class="grid grid-3">${data.featured_matches.length ? data.featured_matches.map(matchCard).join("") : '<div class="empty">Avancez le tournoi pour révéler les prochains matchs.</div>'}</div>
  </div>`;
}

function matchCard(match) {
  const tag = match.prediction_available ? "a" : "div";
  const href = match.prediction_available ? ` href="#/match/${routeId(match.id)}"` : "";
  return `<${tag} class="match-card${match.prediction_available ? "" : " unavailable"}"${href}>
    <div class="meta"><span>${esc(match.round)}</span><span>${match.live_replay_available ? '<i class="live-dot"></i>Replay live' : 'Pré-match'}</span></div>
    <div class="match-player ${match.winner === "A" ? "winner" : ""}"><span>${esc(match.player_a.name)}</span><span>${match.winner === "A" ? "✓" : ""}</span></div>
    <div class="match-player ${match.winner === "B" ? "winner" : ""}"><span>${esc(match.player_b.name)}</span><span>${match.winner === "B" ? "✓" : ""}</span></div>
    ${match.prediction_available ? "" : '<div class="meta">Prédiction indisponible : match hors périmètre Elo</div>'}
  </${tag}>`;
}

async function renderTournaments() {
  const tournaments = await api("/tournaments?limit=60");
  app.innerHTML = `<div class="page"><div class="page-title"><div><div class="eyebrow">Archives interactives</div><h1>Tournois</h1></div></div><div class="grid grid-3">${tournaments.map(tournamentCard).join("")}</div></div>`;
}

async function renderTournament(id) {
  app.innerHTML = '<div class="loading-screen"><div class="ball-loader"></div><p>Simulation du tableau PRE-MATCH V1…</p></div>';
  const [tournament, bracket, predictions, replay] = await Promise.all([
    api(`/tournaments/${routeId(id)}`), api(`/tournaments/${routeId(id)}/bracket`),
    api(`/tournaments/${routeId(id)}/predictions?simulations=2000`), api("/replay")
  ]);
  const active = replay.tournament_id === id;
  const probabilityColumns = [...new Set(predictions.players.flatMap(p => Object.keys(p).filter(k => k.startsWith("P("))))];
  app.innerHTML = `<div class="page">
    <div class="page-title"><div><div class="eyebrow">${esc(tournament.level)} · ${esc(tournament.start_date)} · ${tournament.players} joueurs</div><h1>${esc(tournament.name)}</h1><div class="meta"><span>${esc(tournament.surface)}</span><span>${tournament.matches} matchs</span><span>${tournament.replay_matches} replays point par point</span></div></div><div class="surface-pill" style="background:${surfaceColor(tournament.surface)}">${esc(tournament.surface)}</div></div>
    <section class="replay-banner"><div><div class="eyebrow">${active ? "Session active" : "Archive disponible"}</div><h2>${predictions.next_round ? `Prochain round : ${esc(predictions.next_round)}` : "Tournoi terminé"}</h2><p>${predictions.n_simulations || "—"} simulations · PREMATCH_V1 uniquement · Adapter désactivé</p></div><div class="actions"><button id="select-replay" class="btn">${active ? "Réinitialiser" : "Lancer le replay"}</button><button id="advance-round" class="btn secondary" ${!active || !predictions.next_round ? "disabled" : ""}>Valider le round →</button></div></section>
    <div class="section-head"><div><div class="eyebrow">Tableau réel</div><h2>Bracket</h2></div></div>
    <div class="bracket">${bracket.rounds.filter(r => r.round_order >= 6).map(round => `<section class="bracket-round"><div class="round-heading"><strong>${esc(round.round)}</strong><span>${esc(round.status)}</span></div>${round.matches.map(matchCard).join("")}</section>`).join("")}</div>
    <div class="section-head"><div><div class="eyebrow">Monte Carlo</div><h2>Probabilités de progression</h2></div><span class="badge official">TOURNAMENT_SIM_V1</span></div>
    <div class="card probability-table">${predictions.players.length ? `<table><thead><tr><th>Joueur</th>${probabilityColumns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${predictions.players.slice(0,40).map(player => `<tr><td><strong>${esc(player.player_name)}</strong></td>${probabilityColumns.map(c => `<td class="number">${pct(player[c] || 0)}</td>`).join("")}</tr>`).join("")}</tbody></table>` : '<div class="empty">Tournoi terminé.</div>'}</div>
  </div>`;
  document.querySelector("#select-replay").onclick = async () => {
    await api("/replay/select", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({tournament_id:id})});
    await renderTournament(id);
  };
  document.querySelector("#advance-round").onclick = async () => { await api("/replay/advance-round", {method:"POST"}); await renderTournament(id); };
}

function contributionRows(rows) {
  const max = Math.max(...rows.map(row => Math.abs(row.log_odds_contribution)), .001);
  return rows.map(row => `<div class="contribution"><div><strong>${esc(row.feature.replaceAll("_", " "))}</strong><div class="meta">Valeur ${row.value === null ? "indisponible" : Number(row.value).toFixed(3)}</div></div><div class="contribution-bar"><i style="width:${Math.abs(row.log_odds_contribution)/max*100}%"></i></div></div>`).join("");
}

async function renderMatch(id) {
  const [match, prediction] = await Promise.all([api(`/matches/${routeId(id)}`), api(`/matches/${routeId(id)}/prediction`)]);
  const pa = prediction.probability.player_a;
  app.innerHTML = `<div class="page">
    <div class="match-hero"><div class="eyebrow">${esc(match.tournament_name)} · ${esc(match.round)} · ${esc(match.surface)}</div><div class="versus"><div class="player-side"><div class="rank">Classement #${esc(match.player_a.rank)}</div><h1>${esc(match.player_a.name)}</h1></div><div class="vs">VS</div><div class="player-side"><div class="rank">Classement #${esc(match.player_b.rank)}</div><h1>${esc(match.player_b.name)}</h1></div></div><div class="probability-split" style="--pa:${pa*100}%"><i></i></div><div class="prob-labels"><span>${pct(pa)}</span><span>${pct(1-pa)}</span></div><div class="model-note"><span>Model probability · ${esc(prediction.model_version)}</span><span>Snapshot ${esc(prediction.feature_snapshot_hash.slice(0,10))}</span></div></div>
    <div class="details-layout"><section class="card"><div class="eyebrow">Pourquoi cette probabilité ?</div><h2>Contributions principales</h2>${contributionRows(prediction.top_contributions)}</section><aside class="card"><div class="eyebrow">Contexte</div><h3>${esc(match.score || "Score à venir")}</h3><div class="meta"><span>Format BO${match.best_of}</span><span>${esc(match.duration_minutes)} min</span></div><hr><p>Cette valeur est une probabilité de modèle, pas une prédiction certaine.</p><span class="badge official">${esc(prediction.cache)} cache</span></aside></div>
    <div id="live-root">${match.live_replay_available ? '<div class="loading-screen"><div class="ball-loader"></div><p>Chargement du replay live…</p></div>' : '<div class="empty" style="margin-top:20px">Le point-by-point n’est pas disponible pour ce match. La prédiction pré-match reste consultable.</div>'}</div>
  </div>`;
  if (match.live_replay_available) await initLive(id);
}

async function initLive(id) {
  clearInterval(state.timer); state.timer = null; state.liveHistory = new Map(); state.liveMatch = id;
  await seekLive(0, true);
}

function chartSvg(history) {
  if (!history.length) return "";
  const maxX = Math.max(history.at(-1).point_index, 1);
  const points = history.map(p => `${(p.point_index/maxX)*1000},${(1-p.p_match_a)*200}`).join(" ");
  return `<svg class="chart" viewBox="0 0 1000 200" preserveAspectRatio="none"><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#dc6b42" stop-opacity=".25"/><stop offset="1" stop-color="#dc6b42" stop-opacity="0"/></linearGradient></defs><polygon class="area" points="0,200 ${points} 1000,200"/><polyline points="${points}"/></svg>`;
}

async function seekLive(point, first = false) {
  const data = await api(`/matches/${routeId(state.liveMatch)}/live?point=${point}`);
  state.liveHistory.set(data.point_index, {point_index:data.point_index, p_match_a:data.probabilities.match_a});
  const history = [...state.liveHistory.values()].sort((a,b) => a.point_index-b.point_index);
  const p = data.probabilities, s = data.score;
  document.querySelector("#live-root").innerHTML = `<section class="live-panel"><div class="scoreboard"><div><div class="eyebrow">Historical live replay · point ${data.point_index+1}/${data.total_points}</div><div class="score-player"><span>${s.server === "A" ? '<b class="server-ball">●</b>' : ''}${esc(data.player_a.name)}</span><span class="score-number">${s.sets[0]}</span><span class="score-number">${s.games[0]}</span><span class="score-number score-point">${esc(s.points[0])}</span></div><div class="score-player"><span>${s.server === "B" ? '<b class="server-ball">●</b>' : ''}${esc(data.player_b.name)}</span><span class="score-number">${s.sets[1]}</span><span class="score-number">${s.games[1]}</span><span class="score-number score-point">${esc(s.points[1])}</span></div></div><span class="badge live">Replay</span></div><div class="live-probs">${[["NEXT POINT",p.next_point_a,"POINT_V2"],["GAME",p.game_a,"LIVE_V2"],["SET",p.set_a,"LIVE_V2"],["MATCH",p.match_a,"LIVE_V3"]].map(([label,value,model]) => `<div class="prob-card"><small>${label}</small><strong>${pct(value)}</strong><em>${model} · P(${esc(data.player_a.name.split(" ").at(-1))})</em></div>`).join("")}</div><div class="replay-controls"><button id="play-live" class="btn dark">${state.timer ? "Pause" : "Lecture"}</button><button id="step-live" class="tab">+1 point</button><input id="live-range" type="range" min="0" max="${data.total_points-1}" value="${data.point_index}" aria-label="Position du replay"><span>${data.point_index+1}</span></div><div class="chart-wrap"><div class="eyebrow">Historique P(match) — joueur A</div>${chartSvg(history)}</div></section>`;
  document.querySelector("#live-range").onchange = event => seekLive(Number(event.target.value));
  document.querySelector("#step-live").onclick = () => seekLive(Math.min(data.point_index+1,data.total_points-1));
  document.querySelector("#play-live").onclick = () => {
    if (state.timer) { clearInterval(state.timer); state.timer = null; seekLive(data.point_index); return; }
    let cursor = data.point_index;
    state.timer = setInterval(async () => {
      cursor += 1;
      if (cursor >= data.total_points) { clearInterval(state.timer); state.timer = null; return; }
      await seekLive(cursor);
    }, 800);
    seekLive(data.point_index);
  };
  if (first) {
    const full = await api(`/matches/${routeId(state.liveMatch)}/live/history?through_point=0`);
    full.history.forEach(item => state.liveHistory.set(item.point_index,item));
  }
}

async function renderAdmin() {
  const [metrics, models] = await Promise.all([api("/admin/performance"), api("/models")]);
  const prematch = metrics.prematch.find(row => String(row.model || row.Model || "").toLowerCase().includes("logistic")) || metrics.prematch[0];
  app.innerHTML = `<div class="page"><div class="page-title"><div><div class="eyebrow">Contrôle interne</div><h1>Performance</h1><p>${esc(metrics.prospective_notice)}</p></div></div><div class="metric-grid"><div class="metric"><div class="eyebrow">PRE-MATCH · Log loss</div><strong>${Number(prematch.log_loss || prematch.test_log_loss || 0).toFixed(4)}</strong></div><div class="metric"><div class="eyebrow">Modèles officiels</div><strong>${models.filter(m => m.official).length}</strong></div><div class="metric"><div class="eyebrow">Adapter tournoi</div><strong>Désactivé</strong></div></div><div class="section-head"><div><div class="eyebrow">Registry</div><h2>Versions des moteurs</h2></div></div><div class="card probability-table"><table><thead><tr><th>Version</th><th>Rôle</th><th>Statut</th></tr></thead><tbody>${models.map(model => `<tr><td><strong>${esc(model.model_version)}</strong></td><td>${esc(model.role)}</td><td><span class="badge ${model.official ? "official" : ""}">${model.official ? "officiel" : "recherche"}</span></td></tr>`).join("")}</tbody></table></div></div>`;
}

function upcomingCard(match) {
  const id = routeId(match.external_match_id);
  return `<article class="match-card upcoming-card">
    <a class="upcoming-main-link" href="/matches/${id}">
      <div class="meta"><span>${esc(match.tournament)} · ${esc(match.round)}</span><span>${esc(new Date(match.starts_at).toLocaleString("fr-FR", {dateStyle:"medium", timeStyle:"short"}))}</span></div>
      <div class="match-player"><strong>${esc(match.player_a.name)}</strong><span>A</span></div>
      <div class="match-player"><strong>${esc(match.player_b.name)}</strong><span>B</span></div>
      <div class="meta"><span>${esc(match.surface)}</span><span>${esc(match.venue)}</span></div>
    </a>
    <div class="actions"><button class="btn dark predict-upcoming" data-match-id="${esc(id)}">Actualiser & calculer la prédiction</button></div>
    <div class="prediction-slot"></div>
  </article>`;
}

function predictionResult(prediction) {
  const pa = prediction.probability.player_a;
  const impact = prediction.refresh_impact * 100;
  return `<div class="on-demand-result">
    <div class="probability-split" style="--pa:${pa*100}%"><i></i></div>
    <div class="prob-labels"><span>${esc(prediction.player_a.name)} ${pct(pa)}</span><span>${pct(1-pa)} ${esc(prediction.player_b.name)}</span></div>
    <div class="meta"><span class="badge official">Qualité ${esc(prediction.data_quality)}</span><span>${prediction.feature_count} features · impact refresh ${impact >= 0 ? "+" : ""}${impact.toFixed(1)} pts</span><span>${prediction.external_requests_used} requêtes externes</span></div>
    <p>Données A : ${esc(prediction.freshness.player_a.state_as_of)} · Données B : ${esc(prediction.freshness.player_b.state_as_of)}.</p>
    <details><summary>Limites et fraîcheur</summary><ul>${prediction.warnings.map(item => `<li>${esc(item)}</li>`).join("")}</ul></details>
  </div>`;
}

async function renderUpcoming() {
  const data = await api("/upcoming?days=3");
  app.innerHTML = `<div class="page"><div class="page-title"><div><div class="eyebrow">ATP · calcul explicite uniquement</div><h1>Prochains matchs</h1><p>Ouvrir cette page ne lance aucune prédiction. Chaque bouton rafraîchit uniquement les deux joueurs concernés.</p></div><span class="badge">${esc(data.provider)}</span></div>
    <div class="grid grid-2">${data.matches.length ? data.matches.map(upcomingCard).join("") : '<div class="empty">Aucun match ATP simple identifié sur les trois prochains jours.</div>'}</div></div>`;
  document.querySelectorAll(".predict-upcoming").forEach(button => {
    button.onclick = async () => {
      const card = button.closest(".upcoming-card");
      const slot = card.querySelector(".prediction-slot");
      button.disabled = true;
      slot.innerHTML = '<div class="loading-inline"><div class="ball-loader"></div><span>Résolution des joueurs → résultats récents → 86 features → PREMATCH V1…</span></div>';
      try {
        const prediction = await api(`/upcoming/${button.dataset.matchId}/refresh-prediction`, {method:"POST"});
        slot.innerHTML = predictionResult(prediction);
      } catch (error) {
        slot.innerHTML = `<div class="error compact"><strong>Prédiction non calculée</strong><p>${esc(error.message)}</p></div>`;
      } finally { button.disabled = false; }
    };
  });
}

const metricValue = (value, kind = "number") => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  if (kind === "percent") return pct(value);
  if (kind === "rank") return `#${Math.round(value)}`;
  return Number(value).toFixed(kind === "decimal" ? 0 : 1);
};

function matchupRows(prediction) {
  const a = prediction.player_metrics.player_a;
  const b = prediction.player_metrics.player_b;
  const groups = [
    ["LEVEL", [
      ["ATP ranking", "ranking", "rank"], ["ATP points", "ranking_points", "decimal"],
      ["Elo composite", "elo_composite", "decimal"], ["Age", "age_years", "number"],
    ]],
    ["FORM", [
      ["Win rate · 90d", "win_rate_90d", "percent"], ["Win rate · 365d", "win_rate_365d", "percent"],
      ["Matches · 30d", "matches_30d", "decimal"], ["Rest days", "rest_days", "decimal"],
    ]],
    ["SERVE", [
      ["Service points won · 90d", "service_points_won_rate_90d", "percent"],
      ["1st serve won · 90d", "first_serve_won_rate_90d", "percent"],
      ["2nd serve won · 90d", "second_serve_won_rate_90d", "percent"],
      ["Ace rate · 90d", "ace_rate_90d", "percent"],
    ]],
    ["RETURN", [["Return points won · 90d", "return_points_won_rate_90d", "percent"]]],
    ["CONTEXT", [
      ["Sets · 30d", "sets_30d", "decimal"], ["Games · 30d", "games_30d", "decimal"],
      ["Surface matches", "surface_matches", "decimal"], ["Career matches", "career_matches", "decimal"],
    ]],
  ];
  return groups.map(([group, rows]) => `<tbody><tr class="metric-group"><th colspan="3">${group}</th></tr>${rows.map(([label,key,kind]) => `<tr><td>${esc(label)}</td><td>${metricValue(a[key],kind)}</td><td>${metricValue(b[key],kind)}</td></tr>`).join("")}</tbody>`).join("");
}

function playerAnalysisCard(player, metrics, side) {
  const record = metrics.recent_record || {wins:0, losses:0, sample:0};
  return `<article class="analysis-player ${side}"><div class="side-label">PLAYER ${side.toUpperCase()}</div><h2>${esc(player.name)}</h2>
    <div class="player-kpis"><div><span>ATP</span><strong>${metricValue(metrics.ranking,"rank")}</strong></div><div><span>Elo composite</span><strong>${metricValue(metrics.elo_composite,"decimal")}</strong></div><div><span>Last ${record.sample}</span><strong>${record.wins}–${record.losses}</strong></div></div>
    <div class="meta"><span>${metrics.age_years ? `${Number(metrics.age_years).toFixed(1)} ans` : "Âge indisponible"}</span>${metrics.hand ? `<span>${esc(metrics.hand)}-handed</span>` : ""}</div></article>`;
}

function contributionPanels(prediction) {
  const aRows = prediction.top_contributions.filter(row => row.log_odds_contribution >= 0);
  const bRows = prediction.top_contributions.filter(row => row.log_odds_contribution < 0);
  const rows = items => items.length ? items.map(row => `<li><span>${esc(row.feature.replaceAll("_"," "))}</span><strong>${row.log_odds_contribution >= 0 ? "+" : ""}${Number(row.log_odds_contribution).toFixed(3)}</strong></li>`).join("") : "<li><span>Aucun facteur parmi le top 6</span></li>";
  return `<div class="factor-grid"><div><div class="side-label">FAVORISE ${esc(prediction.player_a.name)}</div><ul>${rows(aRows)}</ul></div><div><div class="side-label">FAVORISE ${esc(prediction.player_b.name)}</div><ul>${rows(bRows)}</ul></div></div><p class="fine-print">Contributions du modèle en log-odds. Elles décrivent son calcul, pas une causalité réelle.</p>`;
}

function predictionHistory(history) {
  if (!history?.length) return "";
  return `<section class="detail-panel"><div class="section-kicker">PREDICTION HISTORY</div><div class="history-list">${history.map(item => `<div><time>${esc(new Date(item.prediction_created_at).toLocaleString("fr-FR", {dateStyle:"short",timeStyle:"short"}))}</time><strong>${pct(item.probability.player_a)}</strong><span>${esc(item.data_quality || "—")}</span></div>`).join("")}</div></section>`;
}

function renderPredictionDetail(match, prediction, history = []) {
  const pa = prediction.probability.player_a;
  const favorite = pa >= .5 ? prediction.player_a.name : prediction.player_b.name;
  const favoriteP = Math.max(pa, 1-pa);
  const hasMetrics = prediction.player_metrics?.player_a && prediction.player_metrics?.player_b;
  const h2h = prediction.features?.h2h_matches_before === null || prediction.features?.h2h_matches_before === undefined ? "" : `<div class="context-chips"><span>H2H antérieurs · ${Math.round(prediction.features.h2h_matches_before)}</span><span>H2H edge A · ${Number(prediction.features.h2h_edge_A || 0).toFixed(3)}</span></div>`;
  const metricsMarkup = hasMetrics ? `<div class="player-analysis-grid">${playerAnalysisCard(prediction.player_a,prediction.player_metrics.player_a,"a")}${playerAnalysisCard(prediction.player_b,prediction.player_metrics.player_b,"b")}</div>
  <section class="detail-panel"><div class="section-kicker">MATCHUP</div><h2>Comparaison des données utilisées</h2><div class="matchup-table"><table><thead><tr><th>Metric</th><th>${esc(prediction.player_a.name)}</th><th>${esc(prediction.player_b.name)}</th></tr></thead>${matchupRows(prediction)}</table></div>${h2h}</section>` : `<section class="detail-panel"><div class="section-kicker">MATCHUP</div><p>Ce snapshot utilise l’ancien schéma d’affichage. Actualisez les données pour exposer la comparaison A/B.</p></section>`;
  const featureMarkup = prediction.features ? `<details class="detail-panel debug-panel"><summary>ADVANCED / DEBUG · Show ${prediction.feature_count} model features</summary><div class="feature-grid">${Object.entries(prediction.features).map(([key,value]) => `<div><code>${esc(key)}</code><span>${value === null ? "null" : Number(value).toFixed(6)}</span></div>`).join("")}</div><p class="fine-print">Snapshot ${esc(prediction.feature_snapshot_hash)}</p></details>` : "";
  return `<section class="prediction-stage">
    <div class="model-line"><span>PRE-MATCH V1</span><span>Data updated ${esc(relativeAge(prediction.refresh_checked_at || prediction.prediction_created_at))} · Quality ${esc(prediction.data_quality)}</span></div>
    <div class="probability-duel"><div><span>${esc(prediction.player_a.name)}</span><strong>${pct(pa)}</strong><small>Probabilité estimée</small></div><div class="duel-vs">VS</div><div><span>${esc(prediction.player_b.name)}</span><strong>${pct(1-pa)}</strong><small>Probabilité estimée</small></div></div>
    <div class="detail-probability-bar" style="--pa:${pa*100}%"><i></i></div>
    <div class="favorite-line"><span>Model favorite</span><strong>${esc(favorite)} · ${pct(favoriteP)}</strong><em>description du modèle, pas une certitude</em></div>
  </section>
  ${metricsMarkup}
  <section class="detail-panel"><div class="section-kicker">WHY THIS PREDICTION?</div><h2>Model contribution</h2>${contributionPanels(prediction)}</section>
  <section class="detail-panel quality-panel"><div><div class="section-kicker">DATA QUALITY</div><h2>${esc(prediction.data_quality)}</h2><p>A · ${esc(prediction.freshness.player_a.state_as_of)} &nbsp; B · ${esc(prediction.freshness.player_b.state_as_of)}</p></div><ul>${prediction.warnings.map(item => `<li>${esc(item)}</li>`).join("")}</ul></section>
  ${predictionHistory(history)}
  <details class="detail-panel"><summary>DATA SOURCES</summary><dl>${Object.entries(prediction.data_sources || {}).map(([key,value]) => `<dt>${esc(key.replaceAll("_"," "))}</dt><dd>${esc(value)}</dd>`).join("")}</dl></details>
  ${featureMarkup}`;
}

function detailHeader(match) {
  const date = new Date(match.scheduled_at);
  return `<a class="back-link" href="/upcoming">← Prochains matchs</a><header class="detail-header"><div><div class="section-kicker">${esc(match.status)} · ${esc(match.provider)}</div><h1>${esc(match.tournament.name)}</h1><p>${esc(match.tournament.round)} · ${esc(match.tournament.surface)}<br>${esc(date.toLocaleString("fr-FR", {dateStyle:"long",timeStyle:"short"}))}</p></div><span class="surface-pill" style="background:${surfaceColor(match.tournament.surface)}">${esc(match.tournament.surface)}</span></header>`;
}

async function renderProspectiveMatch(id) {
  const match = await api(`/matches/${routeId(id)}`);
  const prediction = await apiMaybe(`/matches/${routeId(id)}/prediction`);
  const history = prediction ? await api(`/matches/${routeId(id)}/prediction-history`) : {history:[]};
  app.innerHTML = `<div class="prospective-page"><div class="detail-shell">${detailHeader(match)}<div id="prediction-detail">${prediction ? renderPredictionDetail(match,prediction,history.history) : `<section class="prediction-empty"><div class="section-kicker">PRE-MATCH V1</div><h2>Prediction not calculated</h2><p>Aucun calcul n’est lancé automatiquement. Actualisez les deux joueurs pour créer un snapshot traçable.</p></section>`}</div><div class="detail-actions"><button id="refresh-detail" class="btn lime">${prediction ? "ACTUALISER LES DONNÉES & RECALCULER" : "ACTUALISER & CALCULER"}</button><div id="refresh-status" aria-live="polite"></div></div><section class="future-strip"><span>UPCOMING · actif</span><span>LIVE · préparé</span><span>FINISHED · préparé</span></section></div></div>`;
  document.querySelector("#refresh-detail").onclick = async event => {
    const button = event.currentTarget, status = document.querySelector("#refresh-status");
    const steps = ["Resolving players…","Fetching recent matches…","Updating player A…","Updating player B…","Building 86 features…","Calculating prediction…"];
    let cursor = 0; button.disabled = true; status.textContent = steps[0];
    const timer = setInterval(() => { cursor = Math.min(cursor+1,steps.length-1); status.textContent = steps[cursor]; }, 650);
    try {
      const refreshed = await api(`/matches/${routeId(id)}/refresh-prediction`, {method:"POST"});
      const refreshedHistory = await api(`/matches/${routeId(id)}/prediction-history`);
      document.querySelector("#prediction-detail").innerHTML = renderPredictionDetail(match,refreshed,refreshedHistory.history);
      button.textContent = "ACTUALISER LES DONNÉES & RECALCULER";
      status.textContent = refreshed.cache === "unchanged_features" ? "Données vérifiées · snapshot inchangé" : "Nouveau snapshot calculé";
    } catch (error) {
      status.innerHTML = `<span class="detail-error">Prediction unavailable · ${esc(error.message)}</span>`;
    } finally { clearInterval(timer); button.disabled = false; }
  };
}

async function router() {
  clearInterval(state.timer); state.timer = null;
  const pathParts = location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const hashParts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const parts = ["matches", "upcoming"].includes(pathParts[0]) ? pathParts : hashParts;
  app.innerHTML = '<div class="loading-screen"><div class="ball-loader"></div><p>Chargement…</p></div>';
  try {
    if (!parts.length) await renderHome();
    else if (parts[0] === "tournaments") await renderTournaments();
    else if (parts[0] === "tournament" && parts[1]) await renderTournament(parts[1]);
    else if (parts[0] === "match" && parts[1]) await renderMatch(parts[1]);
    else if (parts[0] === "matches" && parts[1]) await renderProspectiveMatch(parts.slice(1).join("/"));
    else if (parts[0] === "upcoming") await renderUpcoming();
    else if (parts[0] === "admin") await renderAdmin();
    else location.hash = "#/";
    window.scrollTo(0,0);
  } catch (error) {
    console.error(error);
    app.innerHTML = `<div class="error"><div class="eyebrow">Erreur</div><h2>Le court est momentanément indisponible.</h2><p>${esc(error.message)}</p><a class="btn dark" href="#/">Retour à l’accueil</a></div>`;
  }
}

window.addEventListener("hashchange", router);
window.addEventListener("popstate", router);
router();
