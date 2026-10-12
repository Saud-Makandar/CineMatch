import React, { useState, useEffect } from 'react';
import { Search, Star, Sparkles, Loader2, X } from 'lucide-react';

export default function App() {
  const [query, setQuery] = useState('Inception');
  const [suggestions, setSuggestions] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const quickPicks = [
    'Inception',
    'Toy Story',
    'Matrix, The',
    'Fight Club',
    'Interstellar'
  ];

  // Initial recommendation load on startup
  useEffect(() => {
    fetchRecommendations('Inception');
  }, []);

  // Base URL for API requests (supports external backend deployment)
  let apiEnv = import.meta.env.VITE_API_URL || '';
  if (!apiEnv || apiEnv.includes('cinematch-api.onrender.com')) {
    apiEnv = import.meta.env.PROD ? 'https://cinematch-0d4t.onrender.com' : '';
  }
  const API_BASE = apiEnv.replace(/\/+$/, '');

  // Simple autocomplete search
  const handleInputChange = async (e) => {
    const val = e.target.value;
    setQuery(val);
    if (val.trim().length >= 2) {
      try {
        const res = await fetch(`${API_BASE}/api/movies?q=${encodeURIComponent(val)}&limit=5`);
        if (res.ok) {
          const data = await res.json();
          setSuggestions(data.movies || []);
        }
      } catch (err) {
        setSuggestions([]);
      }
    } else {
      setSuggestions([]);
    }
  };

  // Fetch recommendations from simple backend API
  const fetchRecommendations = async (movieName = query) => {
    const titleToSearch = (movieName || query).trim();
    if (!titleToSearch) return;

    setIsLoading(true);
    setErrorMessage('');
    setSuggestions([]);

    try {
      const res = await fetch(`${API_BASE}/api/recommend?movie=${encodeURIComponent(titleToSearch)}&top_k=6`);
      const data = await res.json();

      if (data.status === 'success') {
        setRecommendations(data);
      } else {
        setErrorMessage(data.message || 'Movie not found in the dataset.');
        setRecommendations(null);
      }
    } catch (err) {
      setErrorMessage('Could not connect to the movie recommendation service.');
      setRecommendations(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-badge">
          <Sparkles size={16} />
          KNN Collaborative Filtering
        </div>
        <h1 className="app-title">
          Cine<span>Match</span>
        </h1>
        <p className="app-subtitle">
          Type a movie you like to get instant recommendations.
        </p>
      </header>

      {/* Simple Search Section */}
      <div className="search-card">
        <div className="search-input-wrap">
          <div className="search-field-container">
            <Search size={20} className="search-icon" />
            <input
              type="text"
              className="search-input"
              placeholder="Enter movie title (e.g., Inception, Toy Story, Matrix)..."
              value={query}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  fetchRecommendations();
                }
              }}
            />
            {query && (
              <button
                className="search-clear-btn"
                onClick={() => {
                  setQuery('');
                  setSuggestions([]);
                }}
              >
                <X size={16} />
              </button>
            )}

            {/* Quick Autocomplete dropdown */}
            {suggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {suggestions.map((m, idx) => (
                  <div
                    key={idx}
                    className="suggestion-item"
                    onClick={() => {
                      setQuery(m.title);
                      fetchRecommendations(m.title);
                    }}
                  >
                    <span className="suggestion-title">{m.title}</span>
                    <span className="suggestion-meta">★ {m.avg_rating}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            className="btn-search"
            onClick={() => fetchRecommendations()}
            disabled={isLoading}
          >
            {isLoading ? <Loader2 size={18} className="spin" /> : <Search size={18} />}
            Recommend
          </button>
        </div>

        {/* Quick Picks */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginTop: '16px' }}>
          <span style={{ fontSize: '0.85rem', color: '#64748b', fontWeight: 500 }}>
            Try:
          </span>
          {quickPicks.map((pick, idx) => (
            <button
              key={idx}
              className="quick-pick-btn"
              onClick={() => {
                setQuery(pick);
                fetchRecommendations(pick);
              }}
            >
              {pick}
            </button>
          ))}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="message-box">
          <Loader2 size={28} className="spin" style={{ color: '#4f46e5', margin: '0 auto 10px auto' }} />
          <p>Finding recommendations...</p>
        </div>
      )}

      {/* Error State */}
      {errorMessage && !isLoading && (
        <div className="message-box">
          <p style={{ color: '#ef4444', fontWeight: 600 }}>{errorMessage}</p>
          <p style={{ marginTop: '6px' }}>Try searching another title like "Inception", "Toy Story", or "Matrix, The".</p>
        </div>
      )}

      {/* Single Page Recommendation Results */}
      {!isLoading && recommendations && (
        <div>
          {/* Seed Movie Spotlight */}
          {recommendations.seed_details && (
            <div className="spotlight-box">
              <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: '#4f46e5', fontWeight: 700, letterSpacing: '0.04em' }}>
                Recommendations based on:
              </span>
              <div className="spotlight-title">
                {recommendations.seed_details.title}
              </div>
              <div className="spotlight-meta">
                <span>★ {recommendations.seed_details.avg_rating} / 5.0</span>
                <span>•</span>
                <span>{recommendations.seed_details.num_ratings} ratings</span>
                <span>•</span>
                <span>{recommendations.seed_details.genres}</span>
              </div>
            </div>
          )}

          {/* Simple Cards Grid */}
          <div className="movie-grid">
            {recommendations.recommendations.map((movie, idx) => (
              <div key={idx} className="movie-card">
                <div className="card-top">
                  <div>
                    <h3 className="movie-name">{movie.title}</h3>
                    <div className="movie-year">
                      {movie.year ? `Released in ${movie.year}` : ''}
                    </div>
                  </div>
                  <span className="match-badge">
                    {movie.similarity_percentage}% Match
                  </span>
                </div>

                <div className="rating-row">
                  <Star size={16} fill="#f59e0b" className="rating-star" />
                  <span>{movie.avg_rating}</span>
                  <span className="votes-count">
                    ({movie.num_ratings} votes)
                  </span>
                </div>

                <div className="genre-chips">
                  {movie.genres.split('|').map((g, gIdx) => (
                    <span key={gIdx} className="genre-chip">
                      {g}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
