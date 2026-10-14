package usage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/sessions"
	"github.com/yetone/magpie/internal/settings"
)

func TestCodexConversationOTLP(t *testing.T) {
	root := t.TempDir()
	t.Setenv("CODEX_HOME", root)
	dir := filepath.Join(root, "sessions", "2026", "10", "14")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile(filepath.Join("..", "sessions", "testdata", "traces", "codex-conversation-sanitized.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "rollout-2026-10-14T21-01-00-00000000-0000-4000-8000-000000000001.jsonl")
	if err = os.WriteFile(path, fixture, 0600); err != nil {
		t.Fatal(err)
	}
	reader := sessions.NewTraceReader(time.Date(2026, 10, 14, 13, 0, 0, 0, time.UTC))
	spans := reader.Poll(true)
	if len(spans) == 0 {
		t.Fatal("rollout not read by the production reader")
	}
	if again := reader.Poll(true); len(again) != 0 {
		t.Fatal("unchanged rollout exported again")
	}
	e := newOTelExporter()
	defer e.cancel()
	cfg := settings.OTel{Bodies: true, BodiesWhole: true}
	var records []Record
	for _, span := range spans {
		if !span.Pending {
			records = append(records, sessionRecord(span, cfg))
		}
	}
	wire, err := json.Marshal(e.traces(records))
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []struct {
					Attributes []struct {
						Key   string `json:"key"`
						Value struct {
							String string `json:"stringValue"`
						} `json:"value"`
					} `json:"attributes"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err = json.Unmarshal(wire, &payload); err != nil {
		t.Fatal(err)
	}
	models, roots, tools := 0, 0, 0
	for _, resource := range payload.ResourceSpans {
		for _, scope := range resource.ScopeSpans {
			for _, span := range scope.Spans {
				attrs := map[string]string{}
				for _, a := range span.Attributes {
					attrs[a.Key] = a.Value.String
				}
				if attrs["langfuse.session.id"] != "codex-conversation-fixture" {
					t.Fatal("session association lost")
				}
				switch attrs["langfuse.observation.type"] {
				case "generation":
					models++
					for _, key := range []string{"langfuse.observation.input", "langfuse.observation.output"} {
						var messages []map[string]any
						if json.Unmarshal([]byte(attrs[key]), &messages) != nil {
							t.Fatalf("OTLP body lost message structure: %s", attrs[key])
						}
						for _, m := range messages {
							if m["role"] == nil {
								t.Fatalf("OTLP message without role: %s", attrs[key])
							}
						}
					}
				case "span":
					roots++
				case "tool":
					tools++
				}
			}
		}
	}
	if models != 28 || roots != 2 || tools == 0 || !strings.Contains(string(wire), "tool_call_id") || !strings.Contains(string(wire), "Fixture developer message") {
		t.Fatalf("incomplete OTLP conversation: models=%d roots=%d tools=%d", models, roots, tools)
	}
}

func TestCodexConversationToolSecretsStayStructured(t *testing.T) {
	const secret = "fixture-sensitive-value"
	for _, cfg := range []settings.OTel{{Bodies: true}, {Bodies: true, BodiesWhole: true}, {}} {
		body := `[{"role":"tool","tool_call_id":"call","content":{"api_key":"` + secret + `","result":"visible"}}]`
		record := sessionRecord(sessions.TraceSpan{Agent: "codex", Kind: "tool", Output: body}, cfg)
		if strings.Contains(record.BodyOut, secret) || !cfg.Bodies && record.BodyOut != "" {
			t.Fatal("structured tool output bypassed secret scrubbing or body consent")
		}
		if cfg.Bodies && !strings.Contains(record.BodyOut, "visible") {
			t.Fatal("tool result lost during scrubbing")
		}
	}
}

func TestCodexConversationEncodedToolSecrets(t *testing.T) {
	root := t.TempDir()
	t.Setenv("CODEX_HOME", root)
	dir := filepath.Join(root, "sessions", "2026", "10", "14")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	// Codex serializes function arguments and many tool results as JSON strings.
	// A short secret must still be scrubbed by its field name, not its shape.
	const secret = "sh0rt"
	at := time.Now().UTC().Add(-time.Second)
	var rollout []byte
	for i, row := range []struct {
		typ string
		p   map[string]any
	}{
		{"session_meta", map[string]any{"id": "privacy-session"}},
		{"event_msg", map[string]any{"type": "task_started", "turn_id": "turn"}},
		{"response_item", map[string]any{"type": "function_call", "call_id": "call", "name": "read", "arguments": `{"password":"` + secret + `","query":"visible"}`}},
		{"token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-1"}},
		{"response_item", map[string]any{"type": "function_call_output", "call_id": "call", "output": `{"api_key":"` + secret + `","result":"visible"}`}},
		{"token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-2"}},
		{"event_msg", map[string]any{"type": "task_complete", "turn_id": "turn"}},
	} {
		line, _ := json.Marshal(map[string]any{"timestamp": at.Add(time.Duration(i) * time.Millisecond), "type": row.typ, "payload": row.p})
		rollout = append(append(rollout, line...), '\n')
	}
	if err := os.WriteFile(filepath.Join(dir, "rollout-2026-10-14T00-00-00-privacy.jsonl"), rollout, 0600); err != nil {
		t.Fatal(err)
	}
	for _, config := range []settings.OTel{{Bodies: true}, {Bodies: true, BodiesWhole: true}, {}} {
		reader := sessions.NewTraceReader(at)
		spans := reader.Poll(config.Bodies)
		if len(spans) == 0 {
			t.Fatal("production reader did not read the privacy fixture")
		}
		visible := false
		for _, span := range spans {
			r := sessionRecord(span, config)
			for _, body := range []string{r.BodyIn, r.BodyOut} {
				if strings.Contains(body, secret) || !config.Bodies && body != "" {
					t.Fatalf("tool secret or body consent bypassed: kind=%s body=%s", span.Kind, body)
				}
				visible = visible || strings.Contains(body, "visible")
			}
		}
		if config.Bodies && !visible {
			t.Fatal("ordinary tool parameters and results disappeared")
		}
	}
}
