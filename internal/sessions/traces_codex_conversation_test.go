package sessions

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type codexConversationMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func codexConversationMessages(t *testing.T, body string) []codexConversationMessage {
	t.Helper()
	var messages []codexConversationMessage
	if err := json.Unmarshal([]byte(body), &messages); err != nil {
		t.Fatalf("not a message array: %s: %v", body, err)
	}
	for _, m := range messages {
		if m.Role == "" {
			t.Fatalf("message without role: %s", body)
		}
	}
	return messages
}

func TestCodexConversationResponsesDoNotRepeatOldReplies(t *testing.T) {
	c := traceTestCursor("codex")
	at := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	feed := func(typ string, p any) []TraceSpan {
		at = at.Add(time.Millisecond)
		return traceLine(t, c, at, typ, p, true)
	}
	feed("session_meta", map[string]any{"id": "session", "base_instructions": map[string]string{"text": "System instructions."}})
	feed("event_msg", map[string]any{"type": "task_started", "turn_id": "turn"})
	feed("response_item", map[string]any{"type": "message", "id": "developer", "role": "developer", "content": []any{map[string]string{"type": "input_text", "text": "Keep the instructions."}}})
	feed("turn_context", map[string]any{"turn_id": "turn", "model": "model"})
	feed("response_item", map[string]any{"type": "message", "id": "user-response", "role": "user", "content": []any{map[string]string{"type": "input_text", "text": "Inspect the files."}}})
	feed("event_msg", map[string]any{"type": "item_completed", "item": map[string]any{"type": "UserMessage", "id": "user-ui", "content": []any{map[string]string{"type": "text", "text": "Inspect the files."}}}})
	feed("event_msg", map[string]any{"type": "item_completed", "item": map[string]any{"type": "AgentMessage", "id": "message-1", "content": []any{map[string]string{"type": "Text", "text": "I will inspect the files."}}}})
	feed("response_item", map[string]any{"type": "message", "id": "message-1", "role": "assistant", "content": []any{map[string]string{"type": "output_text", "text": "I will inspect the files."}}})
	feed("response_item", map[string]any{"type": "function_call", "call_id": "call-1", "name": "read", "arguments": `{"path":"a.txt"}`})
	first := feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-1", "usage": map[string]int{"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 5}})[0]
	input := codexConversationMessages(t, first.Input)
	output := codexConversationMessages(t, first.Output)
	if len(input) != 3 || input[0].Role != "system" || input[1].Role != "developer" || input[2].Role != "user" || len(output) != 2 || output[0].Role != "assistant" || len(output[1].ToolCalls) != 1 {
		t.Fatalf("missing instruction, duplicated echo, or missing tool: input=%s output=%s", first.Input, first.Output)
	}
	if strings.Contains(first.Output, `"type":"Text"`) || strings.Contains(first.Output, "output_text") || first.Tokens.Input != 60 || first.Tokens.CacheRead != 40 {
		t.Fatal("native content format or token ownership changed")
	}
	call := output[1].ToolCalls[0]
	if call.ID != "call-1" || call.Function.Name != "read" || call.Function.Arguments != `{"path":"a.txt"}` {
		t.Fatalf("tool call changed: %+v", call)
	}
	tool := feed("response_item", map[string]any{"type": "function_call_output", "call_id": "call-1", "output": "File contents."})[0]
	results := codexConversationMessages(t, tool.Output)
	if len(results) != 1 || results[0].Role != "tool" || results[0].ToolCallID != call.ID || string(results[0].Content) != `"File contents."` {
		t.Fatalf("unpaired tool output: %s", tool.Output)
	}
	// A model can emit only a tool call. It must not inherit the previous text.
	feed("response_item", map[string]any{"type": "custom_tool_call", "call_id": "call-2", "name": "exec", "input": "echo done"})
	second := feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-2"})[0]
	next := codexConversationMessages(t, second.Input)
	if len(next) != 6 || !reflect.DeepEqual(next[:3], input) || !reflect.DeepEqual(next[3:5], output) || next[5].ToolCallID != "call-1" {
		t.Fatalf("conversation continuity lost: %s", second.Input)
	}
	if strings.Contains(second.Output, "I will inspect") || len(codexConversationMessages(t, second.Output)) != 1 {
		t.Fatalf("old assistant reply replayed: %s", second.Output)
	}
	feed("response_item", map[string]any{"type": "custom_tool_call_output", "call_id": "call-2", "output": []any{map[string]string{"type": "text", "text": "Done."}}})
	empty := feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-3"})[0]
	if empty.Output != "[]" || !strings.Contains(empty.Input, "call-2") || !strings.Contains(empty.Input, "Done.") {
		t.Fatalf("empty response reused old content or lost tool result: %+v", empty)
	}
	// Identical text in a later response is still a new, recorded message.
	feed("response_item", map[string]any{"type": "message", "id": "message-2", "role": "assistant", "content": []any{map[string]string{"type": "output_text", "text": "I will inspect the files."}}})
	repeated := feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-4"})[0]
	if len(codexConversationMessages(t, repeated.Output)) != 1 || !strings.Contains(repeated.Output, "I will inspect") {
		t.Fatal("a real repeated message was suppressed")
	}
}

func TestCodexConversationCompletedTurnIsNotRecreated(t *testing.T) {
	for _, bodies := range []bool{false, true} {
		c := traceTestCursor("codex")
		at := time.Now().UTC()
		traceLine(t, c, at, "session_meta", map[string]any{"id": "session"}, bodies)
		traceLine(t, c, at, "event_msg", map[string]any{"type": "task_started", "turn_id": "old"}, bodies)
		traceLine(t, c, at, "event_msg", map[string]any{"type": "task_complete", "turn_id": "old", "last_agent_message": "Old answer."}, bodies)
		traceLine(t, c, at.Add(time.Second), "event_msg", map[string]any{"type": "thread_settings_applied"}, bodies)
		traceLine(t, c, at.Add(time.Second), "event_msg", map[string]any{"type": "token_count"}, bodies)
		spans := traceLine(t, c, at.Add(2*time.Second), "event_msg", map[string]any{"type": "task_started", "turn_id": "new"}, bodies)
		if len(spans) != 1 || !spans[0].Pending || spans[0].Turn != "new" {
			t.Fatalf("completed turn exported again: %+v", spans)
		}
		if spans := traceLine(t, c, at.Add(3*time.Second), "event_msg", map[string]any{"type": "task_complete", "turn_id": "old"}, bodies); len(spans) != 0 {
			t.Fatalf("duplicate completion recreated old turn: %+v", spans)
		}
	}
}

func TestCodexConversationMultiPartUserEcho(t *testing.T) {
	for _, sameID := range []bool{false, true} {
		for _, nativeFirst := range []bool{false, true} {
			c := traceTestCursor("codex")
			at := time.Now().UTC()
			traceLine(t, c, at, "session_meta", map[string]any{"id": "session"}, true)
			traceLine(t, c, at, "event_msg", map[string]any{"type": "task_started", "turn_id": "turn"}, true)
			response := map[string]any{"type": "message", "id": "response-user", "role": "user", "content": []any{map[string]string{"type": "input_text", "text": "Instructions."}, map[string]string{"type": "input_text", "text": "Environment."}, map[string]string{"type": "input_text", "text": "Actual question."}}}
			native := map[string]any{"type": "item_completed", "item": map[string]any{"type": "UserMessage", "id": "native-user", "content": []any{map[string]string{"type": "text", "text": "Instructions."}}}}
			if sameID {
				native["item"].(map[string]any)["id"] = "response-user"
			}
			if nativeFirst {
				traceLine(t, c, at, "event_msg", native, true)
			}
			traceLine(t, c, at, "response_item", response, true)
			if !nativeFirst {
				traceLine(t, c, at, "event_msg", native, true)
			}
			model := traceLine(t, c, at, "token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response"}, true)[0]
			messages := codexConversationMessages(t, model.Input)
			if len(messages) != 1 || !strings.Contains(model.Input, "Actual question.") || !strings.Contains(model.Input, "Environment.") {
				t.Fatalf("partial echo duplicated or replaced the full question: %s", model.Input)
			}
			root := traceLine(t, c, at, "event_msg", map[string]any{"type": "task_complete", "turn_id": "turn"}, true)[0]
			if !strings.Contains(root.Input, "Actual question.") {
				t.Fatalf("root lost the real question: %s", root.Input)
			}
		}
	}
}

func TestCodexConversationRecordedOperationDetails(t *testing.T) {
	c := traceTestCursor("codex")
	at := time.Now().UTC()
	traceLine(t, c, at, "session_meta", map[string]any{"id": "session"}, true)
	traceLine(t, c, at, "event_msg", map[string]any{"type": "task_started", "turn_id": "turn"}, true)
	for _, item := range []map[string]any{
		{"type": "FileChange", "id": "patch", "changes": map[string]any{"fixture.txt": map[string]string{"type": "update", "content": "the diff"}}, "stdout": "patch applied", "stderr": ""},
		{"type": "ImageView", "id": "image", "path": "/workspace/fixture.png"},
		{"type": "Extension", "id": "web", "kind": "web", "query": "the search", "action": map[string]string{"type": "open", "url": "https://example.test"}, "results": []any{map[string]string{"text": "the search result"}}},
	} {
		spans := traceLine(t, c, at, "event_msg", map[string]any{"type": "item_completed", "item": item}, true)
		if len(spans) != 1 || spans[0].Kind != "tool" {
			t.Fatalf("recorded operation lost: %+v", item)
		}
		s := spans[0]
		switch item["type"] {
		case "FileChange":
			if !strings.Contains(s.Input, "the diff") || !strings.Contains(s.Output, "patch applied") {
				t.Fatal("patch parameters or result missing")
			}
		case "ImageView":
			if !strings.Contains(s.Input, "/workspace/fixture.png") {
				t.Fatal("image path missing")
			}
		case "Extension":
			if !strings.Contains(s.Input, "the search") || !strings.Contains(s.Output, "the search result") {
				t.Fatal("extension parameters or result missing")
			}
		}
	}
}

func TestCodexConversationSanitizedRollout(t *testing.T) {
	for _, bodies := range []bool{false, true} {
		f, err := os.Open(filepath.Join("testdata", "traces", "codex-conversation-sanitized.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		c := traceTestCursor("codex")
		var spans []TraceSpan
		s := bufio.NewScanner(f)
		for s.Scan() {
			spans = append(spans, c.line(s.Bytes(), bodies)...)
		}
		f.Close()
		if s.Err() != nil {
			t.Fatal(s.Err())
		}
		_, models, tools := spanKinds(spans)
		if len(models) != 28 || len(tools) == 0 {
			t.Fatalf("model/tool records lost: models=%d tools=%d", len(models), len(tools))
		}
		roots := 0
		previous := map[string][]codexConversationMessage{}
		replies := map[string]int{}
		for _, span := range spans {
			if !bodies && (span.Input != "" || span.Output != "") {
				t.Fatal("body leaked with content export disabled")
			}
			if span.Kind == "span" && !span.Pending {
				roots++
			}
			if !bodies || span.Kind != "generation" {
				continue
			}
			input, output := codexConversationMessages(t, span.Input), codexConversationMessages(t, span.Output)
			users := 0
			for _, message := range input {
				if message.Role == "user" {
					users++
				}
			}
			if users != 1 {
				t.Fatalf("multi-part user input echoed as a second message: users=%d input=%s", users, span.Input)
			}
			before := previous[span.Turn]
			if len(input) < len(before) || len(before) > 0 && !reflect.DeepEqual(input[:len(before)], before) {
				t.Fatalf("history omitted at %s: %s", span.ID, span.Input)
			}
			previous[span.Turn] = append(input, output...)
			for _, message := range output {
				if len(message.Content) != 0 {
					replies[string(message.Content)]++
				}
			}
		}
		if roots != 2 || len(c.turns) != 0 || c.current != "" {
			t.Fatalf("completed turns retained or duplicated: roots=%d turns=%d current=%s", roots, len(c.turns), c.current)
		}
		if bodies {
			if len(replies) != 8 {
				t.Fatalf("recorded assistant messages lost: %+v", replies)
			}
			for reply, count := range replies {
				if count != 1 {
					t.Fatalf("old reply repeated %d times: %s", count, reply)
				}
			}
		}
	}
}

func TestCodexConversationNativeMCPResultIsAuthoritative(t *testing.T) {
	for _, nativeFirst := range []bool{false, true} {
		c := traceTestCursor("codex")
		at := time.Now().UTC()
		feed := func(typ string, payload any) []TraceSpan {
			at = at.Add(time.Millisecond)
			return traceLine(t, c, at, typ, payload, true)
		}
		feed("session_meta", map[string]any{"id": "session"})
		feed("event_msg", map[string]any{"type": "task_started", "turn_id": "turn"})
		feed("response_item", map[string]any{"type": "function_call", "call_id": "mcp", "name": "read", "arguments": `{}`})
		feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-1"})
		native := map[string]any{"type": "item_completed", "item": map[string]any{"type": "McpToolCall", "id": "mcp", "tool": "read", "result": map[string]string{"text": "authoritative recorded result"}}}
		if nativeFirst {
			feed("event_msg", native)
		}
		feed("response_item", map[string]any{"type": "function_call_output", "call_id": "mcp", "output": "inferred raw result"})
		if !nativeFirst {
			feed("event_msg", native)
		}
		model := feed("token_usage_record", map[string]any{"turn_id": "turn", "response_id": "response-2"})[0]
		if !strings.Contains(model.Input, "authoritative recorded result") || strings.Contains(model.Input, "inferred raw result") {
			t.Fatalf("nativeFirst=%t: model history kept the inferred result: %s", nativeFirst, model.Input)
		}
		messages := codexConversationMessages(t, model.Input)
		tools := 0
		for _, m := range messages {
			if m.Role == "tool" {
				tools++
			}
		}
		if tools != 1 {
			t.Fatalf("nativeFirst=%t: tool result echoed %d times", nativeFirst, tools)
		}
	}
}

func TestCodexConversationBodyConsentKeepsMetadata(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("testdata", "traces", "codex-conversation-sanitized.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var reference []TraceSpan
	for _, bodies := range []bool{false, true} {
		c := traceTestCursor("codex")
		var spans []TraceSpan
		for _, line := range strings.Split(string(fixture), "\n") {
			for _, span := range c.line([]byte(line), bodies) {
				if !bodies && (span.Input != "" || span.Output != "") {
					t.Fatal("body leaked with consent disabled")
				}
				span.Input, span.Output = "", ""
				spans = append(spans, span)
			}
		}
		if !bodies {
			reference = spans
		} else if !reflect.DeepEqual(reference, spans) {
			t.Fatal("body export changed IDs, timing, ancestry, errors, or token accounting")
		}
	}
}

func BenchmarkCodexConversationRollout(b *testing.B) {
	fixture, err := os.ReadFile(filepath.Join("testdata", "traces", "codex-conversation-sanitized.jsonl"))
	if err != nil {
		b.Fatal(err)
	}
	lines := strings.Split(string(fixture), "\n")
	for _, bodies := range []bool{false, true} {
		name := "metadata"
		if bodies {
			name = "bodies"
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(fixture)))
			for b.Loop() {
				c := newTraceCursor(file{agent: "codex"})
				for _, line := range lines {
					c.line([]byte(line), bodies)
				}
			}
		})
	}
}
