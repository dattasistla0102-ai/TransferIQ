
import { useEffect, useState } from "react";
import { Routes, Route, useNavigate, useSearchParams } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  CircleDollarSign,
  Crosshair,
  LayoutDashboard,
  LoaderCircle,
  Search,
  Shield,
  Target,
  Users,
  Zap,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const features = [
  { id: "scout", number: "01", title: "PLAYER SCOUT", description: "Discover similar players using KNN.", icon: Crosshair },
  { id: "value", number: "02", title: "MARKET VALUE AI", description: "Predict player market value.", icon: CircleDollarSign },
  { id: "bargain", number: "03", title: "BARGAIN FINDER", description: "Discover potentially cheaper alternatives.", icon: Target },
  { id: "analytics", number: "04", title: "PERFORMANCE LAB", description: "Explore player statistics and charts.", icon: BarChart3 },
];


function PlayerProfilePage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const playerName = searchParams.get("name");
  const team = searchParams.get("team");

  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    async function loadProfile() {
      if (!playerName || !team) {
        setError("Player name or team is missing.");
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError("");

        const response = await fetch(
          `${API_URL}/api/player-profile?player_name=${encodeURIComponent(playerName)}&team=${encodeURIComponent(team)}`
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || data.message || "Could not load player profile."
          );
        }

        setProfile(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadProfile();
  }, [playerName, team]);

  useEffect(() => {
    setImageFailed(false);
  }, [profile?.player_id]);

  const imageUrl =
    profile?.player_id
      ? `https://cdn.sofifa.net/players/${String(profile.player_id).slice(0, 3)}/${String(profile.player_id).slice(3)}/24_120.png`
      : null;

  const categories = profile?.statistics
    ? Object.entries(profile.statistics)
    : [];

  const getStat = (category, name) =>
    profile?.statistics?.[category]?.[name] ?? "—";

  const featuredStats = [
    {
      label: "Appearances",
      value: getStat("Overview", "Appearances"),
      accent: "cyan",
    },
    {
      label: "Goals",
      value: getStat("Attacking", "Goals"),
      accent: "pink",
    },
    {
      label: "Assists",
      value: getStat("Attacking", "Assists"),
      accent: "purple",
    },
    {
      label: "FotMob Rating",
      value: getStat("Overview", "FotMob Rating"),
      accent: "gold",
    },
  ];

  const formatCategory = (category) =>
    category.toUpperCase();

  return (
    <div className="page-content profile-page">
      <button
        className="back-button"
        onClick={() => navigate(-1)}
      >
        <ArrowLeft size={15} />
        BACK
      </button>

      <section className="profile-dashboard">

        {loading ? (
          <div className="profile-state">
            <div className="profile-loader" />
            <h3>Loading Player Profile</h3>
            <p>Retrieving 2023/24 season statistics...</p>
          </div>
        ) : error ? (
          <div className="profile-state profile-error">
            <h3>Unable to load profile</h3>
            <p>{error}</p>
            <button
              className="profile-back-action"
              onClick={() => navigate(-1)}
            >
              Go Back
            </button>
          </div>
        ) : profile ? (
          <>
            {/* PLAYER IDENTITY HEADER */}
            <div className="profile-hero">

              <div className="profile-player-identity">
                <div className="profile-player-image">
                  {imageUrl && !imageFailed ? (
                    <img
                      src={imageUrl}
                      alt={profile.player_name}
                      onError={() => setImageFailed(true)}
                    />
                  ) : (
                    <div className="profile-image-fallback">
                      {profile.player_name
                        ?.split(" ")
                        .map((part) => part[0])
                        .slice(0, 2)
                        .join("")
                        .toUpperCase()}
                    </div>
                  )}
                </div>

                <div className="profile-player-details">
                  <div className="profile-eyebrow">
                    PLAYER INTELLIGENCE
                  </div>

                  <h1>{profile.player_name}</h1>

                  <p className="profile-team">
                    {profile.team}
                  </p>

                  <div className="profile-player-tags">
                    <span>{profile.position || "Position unavailable"}</span>
                    <span>PREMIER LEAGUE</span>
                    <span>2023/24</span>
                  </div>
                </div>
              </div>

              <div className="profile-season-card">
                <span>SEASON DATA</span>
                <strong>23/24</strong>
                <p>Recorded performance</p>
              </div>

            </div>

            {/* FEATURED STATISTICS */}
            <div className="profile-section-heading">
              <div>
                <span className="profile-section-kicker">
                  PERFORMANCE OVERVIEW
                </span>
                <h2>Season at a glance</h2>
              </div>

              <span className="profile-data-label">
                ACTUAL SEASON STATS
              </span>
            </div>

            <div className="profile-featured-grid">
              {featuredStats.map((stat) => (
                <div
                  className={`profile-feature-card accent-${stat.accent}`}
                  key={stat.label}
                >
                  <span>{stat.label}</span>
                  <strong>{stat.value}</strong>
                  <div className="profile-feature-line" />
                </div>
              ))}
            </div>

            {/* ADDITIONAL HIGHLIGHTS */}
            <div className="profile-highlight-grid">
              <div className="profile-highlight-card">
                <span>MINUTES PLAYED</span>
                <strong>
                  {getStat("Overview", "Minutes Played")}
                </strong>
              </div>

              <div className="profile-highlight-card">
                <span>PLAYER OF THE MATCH</span>
                <strong>
                  {getStat("Overview", "Player of the Match Awards")}
                </strong>
              </div>

              <div className="profile-highlight-card">
                <span>EXPECTED GOALS</span>
                <strong>
                  {getStat("Attacking", "Expected Goals (xG)")}
                </strong>
              </div>

              <div className="profile-highlight-card">
                <span>EXPECTED ASSISTS</span>
                <strong>
                  {getStat("Creativity", "Expected Assists (xA)")}
                </strong>
              </div>
            </div>

            {/* DETAILED STATISTICS */}
            <div className="profile-section-heading profile-detail-heading">
              <div>
                <span className="profile-section-kicker">
                  COMPLETE BREAKDOWN
                </span>
                <h2>Performance metrics</h2>
                <p>
                  Recorded statistics across the 2023/24 season.
                </p>
              </div>
            </div>

            <div className="profile-stat-layout">
              {categories
                .filter(([category]) => category !== "Overview")
                .map(([category, stats]) => {
                  const statEntries = Object.entries(stats);

                  return (
                    <div
                      className="profile-stat-panel"
                      key={category}
                    >
                      <div className="profile-stat-panel-heading">
                        <div className="profile-category-indicator" />
                        <h3>{formatCategory(category)}</h3>
                        <span>
                          {statEntries.length} METRICS
                        </span>
                      </div>

                      <div className="profile-stat-list">
                        {statEntries.map(([statName, value]) => (
                          <div
                            className="profile-stat-row"
                            key={statName}
                          >
                            <span>{statName}</span>
                            <strong>{value}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
            </div>

            <div className="profile-footer">
              <span>TRANSFERIQ PLAYER INTELLIGENCE</span>
              <span>DATA SEASON · 2023/24</span>
            </div>
          </>
        ) : null}

      </section>
    </div>
  );
}


function Logo() {
  return (
    <div className="brand">
      <div className="brand-mark">T</div>
      <div><b>TRANSFER<span>IQ</span></b><small>FOOTBALL INTELLIGENCE</small></div>
    </div>
  );
}

function PlayerCard({ player, index }) {
  const navigate = useNavigate();
  const [imageFailed, setImageFailed] = useState(false);
  const attributeEntries = Object.entries(player.attributes || {});

  // Try the SoFIFA image CDN when the matching player ID is available.
  // Fall back to initials if the image cannot be loaded.
  const imageUrl = player.player_id
    ? `https://cdn.sofifa.net/players/${player.player_id.slice(0, 3)}/${player.player_id.slice(3)}/24_120.png`
    : null;

  return (
    <article className="player-card">
      <div className="player-card-head">
        {imageUrl && !imageFailed ? (
          <img
            className="player-photo"
            src={imageUrl}
            alt={player.name}
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="player-photo player-placeholder">
            {player.name.split(/\s+/).map(part => part[0]).slice(0, 2).join("").toUpperCase()}
          </div>
        )}

        <div className="player-identity">
          <span className="similar-label">SIMILAR PLAYER {String(index + 1).padStart(2, "0")}</span>
          <h3>{player.name}</h3>
          <p>{player.team}</p>
          <span className="position-tag">{player.position}</span>
        </div>
      </div>

      <div className="attribute-grid">
        {attributeEntries.map(([name, value]) => (
          <div className="attribute" key={name}>
            <div className="attribute-label">{name}</div>
            <div className="attribute-value">
              {value === null || value === undefined ? "N/A" : value}
            </div>
            <div className="attribute-track">
              <div
                className="attribute-fill"
                style={{
                  width: typeof value === "number"
                    ? `${Math.max(0, Math.min(100, value))}%`
                    : "0%",
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <button
        className="secondary-button"
        onClick={() =>
          navigate(
            `/player?name=${encodeURIComponent(player.name)}&team=${encodeURIComponent(player.team)}`
          )
        }
      >
        PLAYER PROFILE <ArrowRight size={15} />
      </button>
    </article>
  );
}

function App() {
  const [page, setPage] = useState("dashboard");
  const navigate = useNavigate();
  const [playerName, setPlayerName] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState("");
  const [searchedPlayer, setSearchedPlayer] = useState("");
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [marketValueLoading, setMarketValueLoading] = useState(false);
  const [marketPlayerName, setMarketPlayerName] = useState("");
  const [marketPlayer, setMarketPlayer] = useState(null);
  const [marketLoading, setMarketLoading] = useState(false);
  const [marketMessage, setMarketMessage] = useState("");
  const [bargainPlayerName, setBargainPlayerName] = useState("");
  const [bargainLoading, setBargainLoading] = useState(false);
  const [bargainReport, setBargainReport] = useState(null);
  const [bargainMessage, setBargainMessage] = useState("");
  const [performanceData, setPerformanceData] = useState(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceMessage, setPerformanceMessage] = useState("");
  const [activePerformanceMetric, setActivePerformanceMetric] = useState("Top Goal Scorers");

  async function searchPlayers(event) {
    event.preventDefault();

    const query = playerName.trim();

    if (!query) {
      setMessage("Enter a player's name first.");
      return;
    }

    setLoading(true);
    setMessage("");
    setResults([]);
    setSearchedPlayer(query);

    try {
      const response = await fetch(
        `${API_URL}/api/scout?player_name=${encodeURIComponent(query)}`
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.message || data.error || "Player scouting failed."
        );
      }

      const similarPlayers = data.players || [];
      setResults(similarPlayers);

      setMessage(
        similarPlayers.length > 0
          ? `Found ${similarPlayers.length} similar players for ${query}.`
          : "No similar players were returned."
      );
    } catch (error) {
      setMessage(
        error.message.toLowerCase().includes("fetch")
          ? "Cannot connect to the Python backend. Make sure api.py is running."
          : error.message
      );
    } finally {
      setLoading(false);
    }
  }

  async function searchMarketValue(event) {
    event.preventDefault();

    const query = marketPlayerName.trim();

    if (!query) {
      setMarketMessage("Enter a player's name first.");
      return;
    }

    setMarketLoading(true);
    setMarketMessage("");
    setMarketPlayer(null);

    try {
      const response = await fetch(
        `${API_URL}/api/market-value?player_name=${encodeURIComponent(query)}`
      );

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(
          data.error || "Could not retrieve the player's prediction."
        );
      }

      setMarketPlayer(data);
    } catch (error) {
      setMarketMessage(
        error.message.toLowerCase().includes("fetch")
          ? "Cannot connect to the Python backend. Make sure api.py is running."
          : error.message
      );
    } finally {
      setMarketLoading(false);
    }
  }


  async function searchBargain(event) {
    event.preventDefault();

    const query = bargainPlayerName.trim();

    if (!query) {
      setBargainMessage("Please enter a player name first.");
      return;
    }

    setBargainLoading(true);
    setBargainMessage("");
    setBargainReport("");

    try {
      const response = await fetch(
        `${API_URL}/api/bargain?player_name=${encodeURIComponent(query)}`
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || "Bargain Finder search failed.");
      }

      setBargainReport(data);
    } catch (error) {
      setBargainMessage(
        error.message.toLowerCase().includes("fetch")
          ? "Cannot connect to the Python backend. Make sure api.py is running."
          : error.message
      );
    } finally {
      setBargainLoading(false);
    }
  }

  async function loadPerformanceData() {
    setPerformanceLoading(true);
    setPerformanceMessage("");

    try {
      const response = await fetch(`${API_URL}/api/performance`);
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || "Could not load performance data.");
      }

      setPerformanceData(data);
    } catch (error) {
      setPerformanceMessage(
        error.message.toLowerCase().includes("fetch")
          ? "Cannot connect to the Python backend. Make sure api.py is running."
          : error.message
      );
    } finally {
      setPerformanceLoading(false);
    }
  }

  useEffect(() => {
    if (page === "analytics" && !performanceData) {
      loadPerformanceData();
    }
  }, [page]);


  const selectedFeature = features.find(feature => feature.id === page);

  return (
    <Routes>
      <Route path="/player" element={<PlayerProfilePage />} />
      <Route
        path="*"
        element={
          <div className="app-shell">
            <aside className="sidebar">
              <Logo />

              <div className="sidebar-label">WORKSPACE</div>
              <button
                className={`sidebar-link ${page === "dashboard" ? "active" : ""}`}
                onClick={() => setPage("dashboard")}
              >
                <LayoutDashboard size={18} /> Dashboard
              </button>

              <div className="sidebar-label">TRANSFER TOOLS</div>
              {features.map(feature => {
                const Icon = feature.icon;
                return (
                  <button
                    key={feature.id}
                    className={`sidebar-link ${page === feature.id ? "active" : ""}`}
                    onClick={() => {
                      setPage(feature.id);
                      setMessage("");
                      setResults([]);
                    }}
                  >
                    <Icon size={18} />
                    <span>{feature.title}</span>
                    <small>{feature.number}</small>
                  </button>
                );
              })}

              <div className="sidebar-bottom">
                <Zap size={20} />
                <div><b>SEASON DATABASE</b><span>Premier League · 23/24</span></div>
                <i />
              </div>
            </aside>

            <main className="main-area">
              <header className="topbar">
                <div>WORKSPACE <span>/</span> <b>PREMIER LEAGUE</b></div>
                <div className="season-label"><i /> SEASON 2023/24</div>
              </header>

              {page === "dashboard" ? (
                <div className="page-content">
                  <section className="hero">

                    <div className="hero-grid" />

                    <div className="hero-copy">
                      <div className="eyebrow">
                        <i /> THE TRANSFER WINDOW NEVER SLEEPS
                      </div>

                      <h1>
                        SCOUT SMARTER.<br />
                        <em>WIN THE WINDOW.</em>
                      </h1>

                      <p>
                        Your Premier League intelligence hub. Find talent, decode player
                        value, and uncover the signings everyone else missed.
                      </p>

                      <button className="primary-button" onClick={() => setPage("scout")}>
                        ENTER SCOUTING HUB <ArrowRight size={17} />
                      </button>
                    </div>

                    <div className="pitch-decoration">
                      <span>IQ</span>
                      <small>SCOUTING INTELLIGENCE</small>
                    </div>

                  </section>

                  <section className="overview-strip">
                    <div><Users /><span><small>PLAYER DATABASE</small><b>Premier League</b></span></div>
                    <div><Activity /><span><small>ANALYTICS ENGINE</small><b>Performance + Value</b></span></div>
                    <div><Shield /><span><small>TRANSFER TOOLS</small><b>Four scouting modules</b></span></div>
                  </section>

                  <section className="feature-section">
                    <div className="eyebrow"><i /> YOUR COMMAND CENTRE</div>
                    <h2>FOUR TOOLS. <em>ONE EDGE.</em></h2>
                    <div className="feature-grid">
                      {features.map(feature => {
                        const Icon = feature.icon;
                        return (
                          <button className="feature-card" key={feature.id} onClick={() => setPage(feature.id)}>
                            <div className="feature-number">{feature.number}<Icon size={23} /></div>
                            <h3>{feature.title}</h3>
                            <p>{feature.description}</p>
                            <div className="feature-open">OPEN MODULE <ArrowRight size={15} /></div>
                          </button>
                        );
                      })}
                    </div>
                  </section>
                  <footer>TRANSFERIQ © 2026 <span>MADE FOR THE LOVE OF THE GAME</span></footer>
                </div>
              ) : (
                <div className="page-content">
                  <button className="back-button" onClick={() => setPage("dashboard")}>
                    <ArrowLeft size={15} /> BACK TO DASHBOARD
                  </button>

                  <section className="workspace-panel">
                    <div className="eyebrow"><i /> SCOUTING WORKSPACE</div>
                    <h1>{selectedFeature?.title}</h1>
                    <p className="workspace-subtitle">{selectedFeature?.description}</p>

                    {page === "scout" ? (
                      <>
                        <form className="search-form" onSubmit={searchPlayers}>
                          <label htmlFor="player-search">PLAYER NAME</label>
                          <div className="search-row">
                            <div className="search-input">
                              <Search size={19} />
                              <input
                                id="player-search"
                                value={playerName}
                                onChange={event => setPlayerName(event.target.value)}
                                placeholder="e.g. Bruno Fernandes"
                              />
                            </div>
                            <button className="primary-button" type="submit" disabled={loading}>
                              {loading ? <LoaderCircle className="spinner" size={17} /> : <Crosshair size={17} />}
                              {loading ? "SCOUTING..." : "FIND SIMILAR PLAYERS"}
                            </button>
                          </div>
                        </form>

                        {message && (
                          <div className={`status-message ${results.length ? "success" : ""}`}>
                            {message}
                          </div>
                        )}



                        {results.length > 0 && (
                          <section className="results-section">
                            <div className="results-heading">
                              <div>
                                <div className="eyebrow"><i /> SCOUTING RESULTS</div>
                                <h2>Players similar to {searchedPlayer}</h2>
                              </div>
                              <span>{results.length} PLAYERS</span>
                            </div>
                            <div className="players-grid">
                              {results.map((player, index) => (
                                <PlayerCard
                                  key={`${player.name}-${player.team}`}
                                  player={player}
                                  index={index}
                                />
                              ))}
                            </div>
                          </section>
                        )}
                      </>

                    ) : page === "value" ? (
                      <>
                        <form className="search-form" onSubmit={searchMarketValue}>
                          <label htmlFor="market-player-search">PLAYER NAME</label>
                          <div className="search-row">
                            <div className="search-input">
                              <Search size={19} />
                              <input
                                id="market-player-search"
                                value={marketPlayerName}
                                onChange={event => setMarketPlayerName(event.target.value)}
                                placeholder="e.g. Erling Haaland"
                              />
                            </div>
                            <button
                              className="primary-button"
                              type="submit"
                              disabled={marketLoading}
                            >
                              {marketLoading ? (
                                <LoaderCircle className="spinner" size={17} />
                              ) : (
                                <CircleDollarSign size={17} />
                              )}
                              {marketLoading ? "PREDICTING..." : "PREDICT MARKET VALUE"}
                            </button>
                          </div>
                        </form>

                        {marketMessage && (
                          <div className="status-message">{marketMessage}</div>
                        )}

                        {marketPlayer && (
                          <section className="results-section">
                            <div className="results-heading">
                              <div>
                                <div className="eyebrow"><i /> AI VALUATION RESULT</div>
                                <h2>{marketPlayer.player_name}</h2>
                              </div>
                              <span>SEASON {marketPlayer.target_season}</span>
                            </div>

                            <div className="selected-player-value">
                              <span>PREDICTED MARKET VALUE</span>
                              <h2>
                                €{(marketPlayer.predicted_market_value_eur / 1000000).toFixed(2)}M
                              </h2>
                            </div>

                            <div className="attribute-grid">
                              {Object.entries(marketPlayer.attributes || {}).map(
                                ([name, value]) => (
                                  <div className="attribute" key={name}>
                                    <div className="attribute-label">{name}</div>
                                    <div className="attribute-value">
                                      {value === null || value === undefined ? "N/A" : value}
                                    </div>
                                  </div>
                                )
                              )}
                            </div>
                            <button
                              className="secondary-button"
                              onClick={() =>
                                navigate(
                                  `/player?name=${encodeURIComponent(marketPlayer.player_name)}&team=${encodeURIComponent(marketPlayer.club)}`
                                )
                              }
                            >
                              PLAYER PROFILE <ArrowRight size={15} />
                            </button>
                          </section>
                        )}

                      </>
                    ) : page === "bargain" ? (
                      <>
                        <form className="search-form" onSubmit={searchBargain}>
                          <label htmlFor="bargain-player-search">PLAYER NAME</label>

                          <div className="search-row">
                            <div className="search-input">
                              <Search size={19} />
                              <input
                                id="bargain-player-search"
                                value={bargainPlayerName}
                                onChange={event => setBargainPlayerName(event.target.value)}
                                placeholder="e.g. Bruno Fernandes"
                              />
                            </div>

                            <button
                              className="primary-button"
                              type="submit"
                              disabled={bargainLoading}
                            >
                              {bargainLoading ? (
                                <LoaderCircle className="spinner" size={17} />
                              ) : (
                                <Target size={17} />
                              )}
                              {bargainLoading ? "ANALYZING..." : "FIND BARGAINS"}
                            </button>
                          </div>
                        </form>

                        {bargainMessage && (
                          <div className="status-message">{bargainMessage}</div>
                        )}


                        {bargainReport && (
                          <section className="results-section bargain-results">
                            <div className="eyebrow">
                              <i /> BARGAIN ANALYSIS
                            </div>

                            <h2>Transfer Market Comparison</h2>

                            <div className="bargain-target">
                              <div>
                                <span className="bargain-label">YOUR TARGET PLAYER</span>
                                <h3>{bargainReport.target.name}</h3>
                                <p>
                                  {bargainReport.target.team} · {bargainReport.target.position}
                                </p>
                              </div>

                              <div className="bargain-target-value">
                                <span>ESTIMATED VALUE</span>
                                <strong>
                                  €{(bargainReport.target.predicted_value / 1000000).toFixed(2)}M
                                </strong>
                                <small>Model A prediction</small>
                              </div>
                            </div>

                            <button
                              className="secondary-button"
                              onClick={() =>
                                navigate(
                                  `/player?name=${encodeURIComponent(bargainReport.target.name)}&team=${encodeURIComponent(bargainReport.target.team)}`
                                )
                              }
                            >
                              PLAYER PROFILE <ArrowRight size={15} />
                            </button>

                            <h3 className="bargain-section-title">
                              Similar Player Alternatives
                            </h3>

                            {bargainReport.alternatives.length === 0 ? (
                              <p>No eligible alternatives found.</p>
                            ) : (
                              bargainReport.alternatives.map((player, index) => (
                                <div className="bargain-card" key={`${player.name}-${index}`}>
                                  <div className="bargain-card-header">
                                    <div>
                                      <h3>{player.name}</h3>
                                      <p>{player.team} · {player.position}</p>
                                    </div>

                                    <span className="bargain-rank">
                                      ALTERNATIVE {index + 1}
                                    </span>
                                  </div>

                                  <div className="bargain-metrics">
                                    <div>
                                      <span>ESTIMATED VALUE</span>
                                      <strong>
                                        €{(player.predicted_value / 1000000).toFixed(2)}M
                                      </strong>
                                    </div>

                                    <div>
                                      <span>ESTIMATED SAVINGS</span>
                                      <strong>
                                        €{(player.savings / 1000000).toFixed(2)}M
                                      </strong>
                                    </div>

                                    <div>
                                      <span>CHEAPER BY</span>
                                      <strong>{player.cheaper_percent.toFixed(1)}%</strong>
                                    </div>

                                    <div>
                                      <span>KNN DISTANCE</span>
                                      <strong>
                                        {player.knn_distance == null
                                          ? "N/A"
                                          : player.knn_distance.toFixed(3)}
                                      </strong>
                                    </div>
                                  </div>

                                  <div className="bargain-comparison">
                                    <h4>GENERAL STATS</h4>

                                    <div className="bargain-table-wrapper">
                                      <table className="bargain-table">
                                        <thead>
                                          <tr>
                                            <th>PLAYER</th>
                                            <th>PACE</th>
                                            <th>SHOOTING</th>
                                            <th>PASSING</th>
                                            <th>DRIBBLING</th>
                                            <th>DEFENDING</th>
                                            <th>PHYSICAL</th>
                                          </tr>
                                        </thead>

                                        <tbody>
                                          {[bargainReport.target, player].map((p, rowIndex) => (
                                            <tr key={rowIndex}>
                                              <td>{p.name}</td>

                                              {[
                                                "Pace",
                                                "Shooting",
                                                "Passing",
                                                "Dribbling",
                                                "Defending",
                                                "Physical",
                                              ].map((stat) => {
                                                const targetValue =
                                                  bargainReport.target.general_stats?.[stat];
                                                const playerValue = player.general_stats?.[stat];
                                                const value = p.general_stats?.[stat];

                                                let colorClass = "";

                                                if (
                                                  targetValue != null &&
                                                  playerValue != null &&
                                                  targetValue !== playerValue
                                                ) {
                                                  const isBetter =
                                                    rowIndex === 0
                                                      ? targetValue > playerValue
                                                      : playerValue > targetValue;

                                                  colorClass = isBetter
                                                    ? "stat-better"
                                                    : "stat-worse";
                                                }

                                                return (
                                                  <td className={colorClass} key={stat}>
                                                    {value == null ? "N/A" : value.toFixed(0)}
                                                  </td>
                                                );
                                              })}
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>

                                  <div className="bargain-comparison">
                                    <h4>PERFORMANCE STATS</h4>

                                    <div className="bargain-table-wrapper">
                                      <table className="bargain-table">
                                        <thead>
                                          <tr>
                                            <th>PLAYER</th>
                                            <th>APPS</th>
                                            <th>GOALS</th>
                                            <th>ASSISTS</th>
                                            <th>xG</th>
                                            <th>xA</th>
                                            <th>FOTMOB RATING</th>
                                          </tr>
                                        </thead>

                                        <tbody>
                                          {[bargainReport.target, player].map((p, rowIndex) => (
                                            <tr key={rowIndex}>
                                              <td>{p.name}</td>
                                              {[
                                                "Appearances",
                                                "Goals",
                                                "Assists",
                                                "Expected Goals (xG)",
                                                "Expected Assists (xA)",
                                                "FotMob Rating",
                                              ].map((stat) => {
                                                const value = p.performance?.[stat];

                                                return (
                                                  <td key={stat}>
                                                    {value == null
                                                      ? "N/A"
                                                      : value.toFixed(2)}
                                                  </td>
                                                );
                                              })}
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                  <button
                                    className="secondary-button"
                                    onClick={() =>
                                      navigate(
                                        `/player?name=${encodeURIComponent(player.name)}&team=${encodeURIComponent(player.team)}`
                                      )
                                    }
                                  >
                                    PLAYER PROFILE <ArrowRight size={15} />
                                  </button>
                                </div>
                              ))
                            )}
                          </section>
                        )}



                      </>

                    ) : page === "analytics" ? (
                      <section className="performance-lab-dashboard">

                        {performanceLoading ? (
                          <div className="status-message">
                            <LoaderCircle className="spinner" size={18} />
                            Loading performance data...
                          </div>
                        ) : performanceMessage ? (
                          <div className="status-message">{performanceMessage}</div>
                        ) : performanceData ? (() => {
                          const leaderboards = performanceData.leaderboards || {};
                          const metricNames = Object.keys(leaderboards);

                          const activeMetric = leaderboards[activePerformanceMetric]
                            ? activePerformanceMetric
                            : metricNames[0];

                          const players = leaderboards[activeMetric] || [];

                          const maxValue = Math.max(
                            ...players.map(player => Number(player.value) || 0),
                            1
                          );

                          const metricUnits = {
                            "Top Goal Scorers": "Goals",
                            "Top Assist Providers": "Assists",
                            "Highest-Rated Players": "Rating",
                            "Expected Goals (xG)": "xG",
                            "Expected Assists (xA)": "xA",
                            "Successful Dribbles per 90": "Dribbles / 90",
                            "Pass Success Percentage": "Pass %",
                            "Tackles per 90": "Tackles / 90",
                            "Interceptions per 90": "Interceptions / 90",
                            "Clearances per 90": "Clearances / 90",
                          };

                          const openProfile = (player) => {
                            navigate(
                              `/player?name=${encodeURIComponent(player.player_name)}&team=${encodeURIComponent(player.team)}`
                            );
                          };

                          return (
                            <>
                              <div className="performance-lab-header">
                                <div>
                                  <div className="eyebrow">
                                    <i /> PERFORMANCE INTELLIGENCE
                                  </div>
                                </div>
                                <span className="performance-season">
                                  SEASON {performanceData.season}
                                </span>
                              </div>

                              <div className="performance-metric-tabs">
                                {metricNames.map(metric => (
                                  <button
                                    key={metric}
                                    className={activeMetric === metric ? "active" : ""}
                                    onClick={() => setActivePerformanceMetric(metric)}
                                  >
                                    {metric}
                                  </button>
                                ))}
                              </div>

                              <div className="performance-summary-grid">
                                {players.slice(0, 3).map((player, index) => (
                                  <div className="performance-summary-card" key={player.rank}>
                                    <span>TOP PERFORMER · #{player.rank}</span>
                                    <h3>{player.player_name}</h3>
                                    <p>{player.team}</p>
                                    <strong>{Number(player.value).toFixed(2)}</strong>
                                    <small>{metricUnits[activeMetric] || "Value"}</small>
                                  </div>
                                ))}
                              </div>

                              <div className="performance-chart-panel">
                                <div className="performance-panel-heading">
                                  <div>
                                    <span>PLAYER RANKINGS</span>
                                    <h3>{activeMetric}</h3>
                                  </div>
                                  <span>TOP {players.length}</span>
                                </div>

                                <div className="performance-chart-list">
                                  {players.map(player => {
                                    const value = Number(player.value) || 0;
                                    const width = (value / maxValue) * 100;

                                    return (
                                      <div className="performance-chart-row" key={player.rank}>
                                        <span className="performance-rank">
                                          {String(player.rank).padStart(2, "0")}
                                        </span>

                                        <div className="performance-player-info">
                                          <button onClick={() => openProfile(player)}>
                                            {player.player_name}
                                          </button>
                                          <small>{player.team}</small>
                                        </div>

                                        <div className="performance-bar-track">
                                          <div
                                            className="performance-bar-fill"
                                            style={{ width: `${width}%` }}
                                          />
                                        </div>

                                        <strong className="performance-value">
                                          {value.toFixed(2)}
                                        </strong>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>

                              <div className="performance-lab-footer">
                                <span>TRANSFERIQ PERFORMANCE INTELLIGENCE</span>
                                <span>ACTUAL SEASON STATISTICS · 2023/24</span>
                              </div>
                            </>
                          );
                        })() : (
                          <div className="status-message">
                            No performance data available.
                          </div>
                        )}

                      </section>

                    ) : (
                      <div className="coming-soon">


                        <div className="coming-icon">
                          {selectedFeature?.icon && <selectedFeature.icon size={32} />}
                        </div>
                        <h2>Next module to connect</h2>
                        <p>
                          This screen is ready for its interface. We'll connect your existing
                          Python feature next, without retraining or replacing your model.
                        </p>
                        <button
                          className="secondary-button"
                          onClick={() => setPage("dashboard")}
                        >
                          RETURN TO DASHBOARD
                        </button>
                      </div>
                    )}

                  </section>
                  <footer>TRANSFERIQ © 2026 <span>PREMIER LEAGUE DATA LAB</span></footer>
                </div>
              )}
            </main>
          </div>
        }
      />
    </Routes>
  );
}

export default App;
