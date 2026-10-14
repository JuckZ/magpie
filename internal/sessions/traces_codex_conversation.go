package sessions

import (
	"encoding/json"
	"strings"

	"github.com/yetone/magpie/internal/redact"
)

// Codex records messages both as response items and as UI completion events.
// Keep the active turn's transcript, with a separate output boundary for each
// response. A usage record must never reuse the last assistant message.
type codexTraceConversation struct {
	history                            []codexTraceMessage
	output                             []codexTraceMessage
	inputEnd                           int
	seen                               map[string]bool
	calls                              map[string]bool
	results                            map[string]int
	lastKey, lastSrc, lastID, lastRole string
	lastContent                        json.RawMessage
	lastEcho                           bool
}

type codexTraceMessage struct {
	Role       string               `json:"role"`
	Content    json.RawMessage      `json:"content,omitempty"`
	ToolCalls  []codexTraceToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
	Name       string               `json:"name,omitempty"`
}

type codexTraceToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (t *traceTurn) codexConversation() *codexTraceConversation {
	if t.codex == nil {
		t.codex = &codexTraceConversation{inputEnd: -1, seen: map[string]bool{}, calls: map[string]bool{}, results: map[string]int{}}
		if t.codexBase != "" {
			content, _ := json.Marshal(t.codexBase)
			t.codex.message("system", "", "session", content)
		}
	}
	return t.codex
}

// Normalize Codex's input_text/output_text and native Text blocks, retaining
// non-text parts as recorded. Do not open local images or decode opaque data.
func codexTraceContent(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		b, _ := json.Marshal([]map[string]string{{"type": "text", "text": text}})
		return b
	}
	var blocks []json.RawMessage
	if json.Unmarshal(raw, &blocks) != nil {
		return raw
	}
	for i, block := range blocks {
		var part struct{ Type, Text string }
		if json.Unmarshal(block, &part) != nil {
			continue
		}
		switch strings.ToLower(part.Type) {
		case "text", "input_text", "output_text":
			blocks[i], _ = json.Marshal(map[string]string{"type": "text", "text": part.Text})
		}
	}
	b, _ := json.Marshal(blocks)
	return b
}

func (c *codexTraceConversation) append(m codexTraceMessage, output bool) {
	if output {
		if c.inputEnd < 0 {
			c.inputEnd = len(c.history)
		}
		c.output = append(c.output, m)
	}
	c.history = append(c.history, m)
}

func (c *codexTraceConversation) message(role, id, source string, content json.RawMessage) bool {
	content = codexTraceContent(content)
	if len(content) == 0 {
		return false
	}
	m := codexTraceMessage{Role: role, Content: content}
	b, _ := json.Marshal(m)
	key := string(b)
	// IDs agree for assistant echoes; user UI IDs can differ. Only suppress
	// matching adjacent messages from different sources, never repeated text
	// from distinct messages or from later responses.
	richerUser := role == "user" && c.lastRole == "user" && source == "response" && c.lastSrc == "native" && id == c.lastID
	if id != "" && c.seen[role+":"+id] && !richerUser {
		return false
	}
	if id != "" {
		c.seen[role+":"+id] = true
	}
	echo := key == c.lastKey && (role == "user" || id == "" || c.lastID == "")
	if role == "user" && c.lastRole == "user" {
		if source == "native" && c.lastSrc == "response" {
			echo = echo || codexTraceUserEcho(content, c.lastContent)
		} else if source == "response" && c.lastSrc == "native" && codexTraceUserEcho(c.lastContent, content) {
			// If the richer response item arrives second, replace its UI echo
			// rather than retaining two representations of the user's input.
			c.history[len(c.history)-1] = m
			c.lastKey, c.lastSrc, c.lastID = key, source, id
			c.lastContent, c.lastEcho = content, true
			return true
		}
	}
	if !c.lastEcho && source != c.lastSrc && echo {
		c.lastEcho = true
		return false
	}
	c.lastKey, c.lastSrc, c.lastID, c.lastRole = key, source, id, role
	c.lastContent, c.lastEcho = content, false
	c.append(m, role == "assistant")
	return true
}

// A native UserMessage can echo only part of the response item's input, which
// also holds environment text and attachments. Keep the richer recorded input.
func codexTraceUserEcho(echo, input json.RawMessage) bool {
	var parts, full []json.RawMessage
	if json.Unmarshal(echo, &parts) != nil || json.Unmarshal(input, &full) != nil || len(parts) == 0 {
		return false
	}
	next := 0
	for _, part := range full {
		if string(part) == string(parts[next]) {
			next++
			if next == len(parts) {
				return true
			}
		}
	}
	return false
}

func (c *codexTraceConversation) call(id, name string, raw json.RawMessage, output bool) {
	if id == "" || c.calls[id] {
		return
	}
	c.calls[id] = true
	c.lastKey, c.lastSrc, c.lastContent = "", "", nil
	arguments := string(raw)
	var text string
	if json.Unmarshal(raw, &text) == nil {
		arguments = text
	}
	if !json.Valid([]byte(arguments)) {
		b, _ := json.Marshal(map[string]string{"input": arguments})
		arguments = string(b)
	}
	// Arguments must be a string in the message format. Scrub their original
	// JSON before encoding it as a string, so even short secret-named values
	// remain covered by the shared exporter scrubbing rules.
	arguments = string(redact.ScrubJSON([]byte(arguments)))
	call := codexTraceToolCall{ID: id, Type: "function"}
	call.Function.Name, call.Function.Arguments = name, arguments
	c.append(codexTraceMessage{Role: "assistant", ToolCalls: []codexTraceToolCall{call}}, output)
}

func codexTraceResult(id, name, output string) codexTraceMessage {
	// A tool can return JSON, text, or a list of content blocks. Preserve its
	// structure for secret-field scrubbing; the role and call ID pair it.
	b := json.RawMessage(codexTraceJSON(json.RawMessage(output)))
	if !json.Valid(b) {
		b, _ = json.Marshal(output)
	}
	return codexTraceMessage{Role: "tool", ToolCallID: id, Name: name, Content: b}
}

func (c *codexTraceConversation) result(id, name, output string) string {
	m := codexTraceResult(id, name, output)
	if index, exists := c.results[id]; exists {
		// A native MCP completion can follow its inferred raw result. Its
		// recorded result is authoritative in later model inputs too.
		c.history[index] = m
	} else {
		c.results[id] = len(c.history)
		c.lastKey, c.lastSrc, c.lastContent = "", "", nil
		c.append(m, false)
	}
	b, _ := json.Marshal([]codexTraceMessage{m})
	return string(b)
}

// Codex often encodes a JSON object or array inside a JSON string. Preserve
// that structure for the exporter's secret-field scrubber; ordinary text
// stays in the format it was recorded in.
func codexTraceJSON(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		trimmed := strings.TrimSpace(text)
		if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') && json.Valid([]byte(text)) {
			return text
		}
	}
	return string(raw)
}

func (c *codexTraceConversation) response() (string, string) {
	end := c.inputEnd
	if end < 0 {
		end = len(c.history)
	}
	input := "[]"
	if end > 0 {
		b, _ := json.Marshal(c.history[:end])
		input = string(b)
	}
	output := "[]"
	if len(c.output) > 0 {
		b, _ := json.Marshal(c.output)
		output = string(b)
	}
	c.output, c.inputEnd = nil, -1
	// Echo matching is local to a response. Identical text in a later model
	// response is a real message, even if its ID was not recorded.
	c.lastKey, c.lastSrc, c.lastID = "", "", ""
	c.lastContent, c.lastEcho = nil, false
	return input, output
}
