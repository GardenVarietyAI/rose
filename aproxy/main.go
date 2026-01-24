package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

type Config struct {
	ListenAddr     string
	ProbeTimeout   time.Duration
	RequestTimeout time.Duration
	Backends       []string
}

type Proxy struct {
	cfg    Config
	client *http.Client
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	p := &Proxy{
		cfg: cfg,
		client: &http.Client{
			Timeout: cfg.RequestTimeout,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   5 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				MaxIdleConns:          100,
				MaxIdleConnsPerHost:   25,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   5 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
			},
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/v1/chat/completions", p.handleChatCompletions)

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      withLogging(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("listening on %s with %d backends", cfg.ListenAddr, len(cfg.Backends))

	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server error: %v", err)
	}
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start))
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("content-type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true}`))
}

func (p *Proxy) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	_ = r.Body.Close()

	backend, err := p.pickBackendWithFreeSlot(r.Context())
	if err != nil {
		http.Error(w, "no available backend (no free slots)", http.StatusServiceUnavailable)
		return
	}

	p.proxyRequest(w, r, backend, bodyBytes)
}

func (p *Proxy) proxyRequest(w http.ResponseWriter, r *http.Request, backendBase string, body []byte) {
	upstreamURL := strings.TrimRight(backendBase, "/") + "/v1/chat/completions"

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "failed to create upstream request", http.StatusBadGateway)
		return
	}

	copyHeaders(req.Header, r.Header)
	req.Header.Del("Host")
	req.ContentLength = int64(len(body))

	resp, err := p.client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}

	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (p *Proxy) pickBackendWithFreeSlot(ctx context.Context) (string, error) {
	ctx2, cancel := context.WithTimeout(ctx, p.cfg.ProbeTimeout)
	defer cancel()

	type result struct {
		backend string
		ok      bool
	}

	ch := make(chan result, len(p.cfg.Backends))

	for _, b := range p.cfg.Backends {
		go func(backend string) {
			ch <- result{backend: backend, ok: p.backendHasFreeSlot(ctx2, backend)}
		}(b)
	}

	var available []string
	for range p.cfg.Backends {
		r := <-ch
		if r.ok {
			available = append(available, r.backend)
		}
	}

	if len(available) == 0 {
		return "", errors.New("no free slot")
	}

	return available[rand.Intn(len(available))], nil
}

func (p *Proxy) backendHasFreeSlot(ctx context.Context, backendBase string) bool {
	u := strings.TrimRight(backendBase, "/") + "/slots?fail_on_no_slot=1"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return false
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

func copyHeaders(dst http.Header, src http.Header) {
	hopByHop := map[string]bool{
		"Connection":          true,
		"Proxy-Connection":    true,
		"Keep-Alive":          true,
		"Proxy-Authenticate":  true,
		"Proxy-Authorization": true,
		"Te":                  true,
		"Trailer":             true,
		"Transfer-Encoding":   true,
		"Upgrade":             true,
	}
	for k, vv := range src {
		if hopByHop[k] {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func loadConfig() (Config, error) {
	cfg := Config{
		ListenAddr:     envStr("LISTEN_ADDR", ":8000"),
		ProbeTimeout:   envDur("PROBE_TIMEOUT", 250*time.Millisecond),
		RequestTimeout: envDur("REQUEST_TIMEOUT", 10*time.Minute),
	}

	raw := strings.TrimSpace(os.Getenv("BACKENDS"))
	if raw == "" {
		return cfg, errors.New("BACKENDS is required (comma-separated URLs)")
	}

	for _, b := range strings.Split(raw, ",") {
		b = strings.TrimSpace(b)
		if b == "" {
			continue
		}
		if _, err := url.ParseRequestURI(b); err != nil {
			return cfg, errors.New("invalid backend url: " + b)
		}
		cfg.Backends = append(cfg.Backends, b)
	}

	if len(cfg.Backends) == 0 {
		return cfg, errors.New("no backends configured")
	}

	return cfg, nil
}

func envStr(k, def string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return def
}

func envDur(k string, def time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}
