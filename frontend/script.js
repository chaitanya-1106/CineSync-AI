document.addEventListener('DOMContentLoaded', () => {

    /* ── DOM refs ──────────────────────────────────────── */
    const form            = document.getElementById('prediction-form');
    const plotInput       = document.getElementById('plot-input');
    const predictBtn      = document.getElementById('predict-btn');
    const btnText         = document.getElementById('btn-text');
    const predResults     = document.getElementById('prediction-results');
    const recsContainer   = document.getElementById('recommendations-container');

    const metricAccuracy  = document.getElementById('metric-accuracy');
    const metricSamples   = document.getElementById('metric-samples');
    const metricVocab     = document.getElementById('metric-vocab');
    const metricHamming   = document.getElementById('metric-hamming');
    const metricF1Micro   = document.getElementById('metric-f1micro');
    const metricGenres    = document.getElementById('metric-genres');
    const genreBars       = document.getElementById('genre-bars');

    const navbar          = document.getElementById('navbar');
    const navLinks        = document.querySelectorAll('.nav-link');
    const hamburger       = document.getElementById('hamburger');
    const navLinksMenu    = document.getElementById('nav-links');

    /* ── Navbar: scroll shadow ─────────────────────────── */
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 10);
        updateActiveLink();
    });

    /* ── Navbar: hamburger toggle ──────────────────────── */
    hamburger.addEventListener('click', () => {
        navLinksMenu.classList.toggle('open');
    });

    /* ── Navbar: smooth scroll + active link ───────────── */
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
            navLinksMenu.classList.remove('open');
        });
    });

    function updateActiveLink() {
        const sections = ['home', 'predict', 'recommend', 'recent', 'metrics'];
        let current = 'home';
        sections.forEach(id => {
            const el = document.getElementById(id);
            if (el && window.scrollY >= el.offsetTop - 120) {
                current = id;
            }
        });
        navLinks.forEach(link => {
            link.classList.toggle('active', link.dataset.section === current);
        });
    }

    /* ── Load metrics + recent films on startup ────────── */
    fetchMetrics();
    /* ── Movie Search Functionality ──────────────────────── */
    const searchBtn = document.getElementById('movie-search-btn');
    const searchInput = document.getElementById('movie-search-input');
    const searchResults = document.getElementById('movie-search-results');

    searchBtn.addEventListener('click', async () => {
        const query = searchInput.value.trim();
        if (!query) return;
        
        searchBtn.disabled = true;
        searchBtn.textContent = 'Searching...';
        searchResults.innerHTML = `<div class="empty-state centered"><p>Scanning film archives...</p></div>`;
        
        try {
            const res = await fetch(`/api/search?title=${encodeURIComponent(query)}`);
            if (!res.ok) throw new Error('Search lookup failed');
            const data = await res.json();
            
            renderSearchResults(data.recommendations || []);
        } catch (err) {
            console.error(err);
            searchResults.innerHTML = `<div class="empty-state centered"><p style="color:var(--error)">❌ Error: ${err.message}</p></div>`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.textContent = 'Search';
        }
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchBtn.click();
        }
    });

    function renderSearchResults(movies) {
        if (!movies || movies.length === 0) {
            searchResults.innerHTML = `<div class="empty-state centered"><span class="empty-icon">🎞️</span><p>No matching films discovered.</p></div>`;
            return;
        }
        searchResults.innerHTML = movies.map((m, i) => {
            const posterHtml = m.poster
                ? `<div class="movie-poster-wrap"><img class="movie-poster" src="${m.poster}" alt="${m.title} poster" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\'poster-fallback\'><span>🎬</span><span>No Poster</span></div>'"></div>`
                : `<div class="poster-fallback"><span>🎬</span><span>No Poster</span></div>`;
            return `
            <div class="movie-card" style="animation-delay:${i * 60}ms">
                ${posterHtml}
                <div class="movie-title-row">
                    <div>
                        <div class="movie-title">${m.title}</div>
                        ${m.year ? `<div class="movie-year">${m.year}</div>` : ''}
                    </div>
                    <span class="movie-rating">★ ${m.rating}</span>
                </div>
                <div class="movie-genres">
                    ${(m.genres || []).map(g => `<span class="genre-chip">${g}</span>`).join('')}
                </div>
                <div class="movie-overview">${m.overview}</div>
            </div>`;
        }).join('');
    }

    fetchRecentFilms();

    /* ── Form submission ───────────────────────────────── */
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const description = plotInput.value.trim();
        if (!description || description.length < 10) {
            alert('Please enter a longer plot description (at least a sentence).');
            return;
        }

        // Loading state
        btnText.textContent = '⏳ Predicting…';
        predictBtn.disabled = true;
        predResults.innerHTML = `<div class="empty-state"><span class="empty-icon">⚙️</span><p>Running model…</p></div>`;
        recsContainer.innerHTML = `<div class="empty-state centered"><span class="empty-icon">🔄</span><p>Finding recommendations…</p></div>`;

        try {
            /* 1. Predict */
            const predRes = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description })
            });
            if (!predRes.ok) {
                const err = await predRes.json();
                throw new Error(err.detail || 'Prediction failed');
            }
            const predData = await predRes.json();
            renderGenres(predData.genres);

            /* 2. Scroll smoothly to recommend */
            document.getElementById('recommend').scrollIntoView({ behavior: 'smooth' });

            /* 3. Recommend */
            const genreNames = predData.genres.map(g => g.genre);
            const recRes = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ genres: genreNames })
            });
            if (!recRes.ok) throw new Error('Recommendations failed');
            const recData = await recRes.json();
            renderRecommendations(recData.recommendations);

        } catch (err) {
            console.error(err);
            predResults.innerHTML = `<p style="color:var(--error);padding:10px">❌ ${err.message}</p>`;
            recsContainer.innerHTML = `<div class="empty-state centered"><p>Could not load recommendations.</p></div>`;
        } finally {
            btnText.textContent = '✨ Predict Genres';
            predictBtn.disabled = false;
        }
    });

    /* ── Render genre tags ─────────────────────────────── */
    function renderGenres(genres) {
        if (!genres || genres.length === 0) {
            predResults.innerHTML = `<p style="color:var(--text-2)">No genres found above threshold.</p>`;
            return;
        }
        const tagsHtml = genres.map((g, i) => `
            <div class="genre-tag" style="animation-delay:${i * 60}ms">
                <span>${g.genre}</span>
                <span class="conf">${(g.confidence * 100).toFixed(1)}% match</span>
            </div>`).join('');
        predResults.innerHTML = `<div class="tags-container">${tagsHtml}</div>`;
    }

    /* ── Render movie cards (with poster thumbnail) ────── */
    function renderRecommendations(movies) {
        if (!movies || movies.length === 0) {
            recsContainer.innerHTML = `<div class="empty-state centered"><span class="empty-icon">🎞️</span><p>No recommendations found for these genres.</p></div>`;
            return;
        }
        recsContainer.innerHTML = movies.map((m, i) => {
            const posterHtml = m.poster
                ? `<div class="movie-poster-wrap"><img class="movie-poster" src="${m.poster}" alt="${m.title} poster" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\'poster-fallback\'><span>🎬</span><span>No Poster</span></div>'"></div>`
                : `<div class="poster-fallback"><span>🎬</span><span>No Poster</span></div>`;
            return `
            <div class="movie-card" style="animation-delay:${i * 60}ms">
                ${posterHtml}
                <div class="movie-title-row">
                    <div>
                        <div class="movie-title">${m.title}</div>
                        ${m.year ? `<div class="movie-year">${m.year}</div>` : ''}
                    </div>
                    <span class="movie-rating">★ ${m.rating}</span>
                </div>
                <div class="movie-desc">${m.overview || 'No synopsis available.'}</div>
                <div class="movie-genre-tags">
                    ${(m.genres || []).map(g => `<span class="movie-genre-pill">${g}</span>`).join('')}
                </div>
            </div>`;
        }).join('');
    }

    /* ── Fetch + render metrics ────────────────────────── */
    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics');
            if (!res.ok) return;
            const d = await res.json();

            if (d.subset_accuracy != null) metricAccuracy.textContent = (d.subset_accuracy * 100).toFixed(1) + '%';
            if (d.train_size      != null) metricSamples.textContent  = d.train_size.toLocaleString();
            if (d.vocab_size      != null) metricVocab.textContent    = d.vocab_size.toLocaleString();
            if (d.hamming_loss    != null) metricHamming.textContent  = d.hamming_loss.toFixed(4);
            if (d.f1_micro        != null) metricF1Micro.textContent  = (d.f1_micro * 100).toFixed(1) + '%';
            if (d.num_genres      != null) metricGenres.textContent   = d.num_genres;

            // Per-genre bars
            if (d.per_genre) {
                const sorted = Object.entries(d.per_genre).sort((a, b) => b[1].f1 - a[1].f1);
                genreBars.innerHTML = sorted.map(([genre, stats]) => `
                    <div class="genre-bar-row">
                        <span class="genre-bar-label">${genre}</span>
                        <div class="genre-bar-track">
                            <div class="genre-bar-fill" data-f1="${stats.f1}"></div>
                        </div>
                        <span class="genre-bar-pct">${(stats.f1 * 100).toFixed(1)}%</span>
                    </div>`).join('');

                // Animate bars after render
                requestAnimationFrame(() => {
                    document.querySelectorAll('.genre-bar-fill').forEach(bar => {
                        const f1 = parseFloat(bar.dataset.f1) * 100;
                        bar.style.width = f1 + '%';
                    });
                });
            }
        } catch (err) {
            console.error('Metrics load failed:', err);
        }
    }

    /* ── Fetch + render recent films ───────────────────── */
    async function fetchRecentFilms() {
        const container = document.getElementById('recent-films-container');
        try {
            const res = await fetch('/api/recent-films');
            if (!res.ok) throw new Error('Failed to load recent films');
            const data = await res.json();
            const films = data.films || [];

            if (!films.length) {
                container.innerHTML = `<div class="empty-state centered"><span class="empty-icon">🎞️</span><p>No recent films available right now.</p></div>`;
                return;
            }

            container.innerHTML = films.map((m, i) => {
                const posterBlock = m.poster
                    ? `<img class="recent-film-poster" src="${m.poster}" alt="${m.title} poster" loading="lazy"
                           onerror="this.parentElement.innerHTML='<div class=\'recent-film-fallback\'><span>🎬</span><small>No poster</small></div>'"/>`
                    : `<div class="recent-film-fallback"><span>🎬</span><small>No poster</small></div>`;

                const genrePills = (m.genres || []).slice(0, 3).map(g =>
                    `<span class="recent-film-genre-pill">${g}</span>`
                ).join('');

                return `
                <div class="recent-film-card" style="animation-delay:${i * 50}ms" title="${m.overview || ''}">
                    <div class="recent-film-poster-wrap">
                        ${posterBlock}
                        <span class="recent-film-rating-badge">★ ${m.rating}</span>
                    </div>
                    <div class="recent-film-info">
                        <div class="recent-film-title">${m.title}</div>
                        ${m.year ? `<div class="recent-film-year">${m.year}</div>` : ''}
                        <div class="recent-film-genres">${genrePills}</div>
                    </div>
                </div>`;
            }).join('');
        } catch (err) {
            console.error('Recent films load failed:', err);
            container.innerHTML = `<div class="empty-state centered"><span class="empty-icon">⚠️</span><p>Could not load recent films.</p></div>`;
        }
    }

});
