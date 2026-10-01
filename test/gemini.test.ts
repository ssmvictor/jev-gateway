import { describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { NO_TOOL } from "../src/questions.js";
import { readUsage } from "../src/usage.js";
import { fakeJev, fakeUpstream, settled, testConfig } from "./helpers.js";

const geminiRequest = (extra: Record<string, unknown> = {}) => ({
  contents: [
    {
      role: "user",
      parts: [{ text: "what does main.py do?" }],
    },
    {
      role: "model",
      parts: [
        {
          functionCall: {
            name: "shell",
            args: { command: "ls" },
          },
        },
      ],
    },
    {
      role: "user",
      parts: [
        {
          functionResponse: {
            name: "shell",
            response: { output: "main.py\nREADME.md" },
          },
        },
      ],
    },
  ],
  systemInstruction: {
    parts: [{ text: "You are a helpful coding assistant." }],
  },
  tools: [
    {
      functionDeclarations: [
        {
          name: "shell",
          description: "Runs a shell command.",
          parameters: {
            type: "object",
            properties: { command: { type: "string" } },
            required: ["command"],
          },
        },
      ],
    },
  ],
  ...extra,
});

function setup(canned: Parameters<typeof fakeJev>[0]) {
  const jev = fakeJev(canned);
  const upstream = fakeUpstream();
  const app = createApp({ config: testConfig(), askJev: jev.askJev, fetch: upstream.fetchImpl });
  const post = (body: unknown, path = "/v1beta/models/gemini-2.0-flash:generateContent") =>
    app.request(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  return { post, jev, upstream, app };
}

const shellDecision = { tool: { choice: "shell" }, needs_tool: { noul: 0.95 } };

const listPlansTool = {
  functionDeclarations: [{ name: "list_plans", parameters: { type: "object", properties: {} } }],
};
const listPlansDecision = { tool: { choice: "list_plans" }, needs_tool: { noul: 0.95 } };
const listPlansContent = { role: "user", parts: [{ text: "list plans" }] };
const invalidCollectionCases: [string, unknown][] = [
  [
    "contents",
    geminiRequest({ contents: [null, "bad", [], listPlansContent], tools: [listPlansTool] }),
  ],
  [
    "parts",
    geminiRequest({ contents: [{ role: "user", parts: [null, "bad", [], { text: "list plans" }] }], tools: [listPlansTool] }),
  ],
  [
    "systemInstruction.parts",
    geminiRequest({ systemInstruction: { parts: [null, "bad", [], { text: "instructions" }] }, tools: [listPlansTool] }),
  ],
  ["tools", geminiRequest({ tools: [null, "bad", [], listPlansTool] })],
  [
    "functionDeclarations",
    geminiRequest({ tools: [{ functionDeclarations: [null, "bad", [], listPlansTool.functionDeclarations[0]] }] }),
  ],
  [
    "part text",
    geminiRequest({ contents: [{ role: "user", parts: [{ text: 17 }, { text: "list plans" }] }], tools: [listPlansTool] }),
  ],
  [
    "functionCall name",
    geminiRequest({
      contents: [{ role: "model", parts: [{ functionCall: { name: 17 } }, { text: "list plans" }] }],
      tools: [listPlansTool],
    }),
  ],
  [
    "functionResponse",
    geminiRequest({
      contents: [{ role: "user", parts: [{ functionResponse: { name: 17 } }, { text: "list plans" }] }],
      tools: [listPlansTool],
    }),
  ],
  [
    "functionDeclaration name",
    geminiRequest({ tools: [{ functionDeclarations: [{ name: 17 }, listPlansTool.functionDeclarations[0]] }] }),
  ],
];

describe("malformed Gemini collections", () => {
  it.each(invalidCollectionCases)("passes through the original request when %s contains malformed entries", async (_name, body) => {
    const { post, jev, upstream } = setup(listPlansDecision);
    const response = await post(body);

    expect(jev.requests).toHaveLength(0);
    expect(response.headers.get("x-jev-gateway-reason")).toBe("unreadable_request");
    expect(upstream.calls).toHaveLength(1);
    expect(upstream.calls[0]?.body).toEqual(body);
  });

  it("keeps valid hosted tools and unknown multimodal part keys routable", async () => {
    const body = geminiRequest({
      contents: [
        {
          role: "user",
          parts: [
            { text: "list plans" },
            { inlineData: { mimeType: "image/png", data: "AA==" } },
            { futurePart: { value: true } },
          ],
        },
      ],
      tools: [listPlansTool, { googleSearch: {} }, { futureHostedTool: { enabled: true } }],
    });
    const { post, jev, upstream } = setup(listPlansDecision);
    const response = await post(body);

    expect(jev.requests).toHaveLength(1);
    expect(upstream.calls).toHaveLength(0);
    expect(response.status).toBe(200);
  });
});

describe("POST /v1beta/models/...:generateContent", () => {
  it("translates Gemini contents and systemInstruction into Jev turns and tool declarations", async () => {
    const { post, jev } = setup(shellDecision);
    await post(geminiRequest());

    const { state } = jev.requests[0]!;
    expect(state).toEqual({
      assistant_instructions: "You are a helpful coding assistant.",
      conversation: [
        { role: "user", text: "what does main.py do?" },
        { role: "assistant", tool_calls: [{ tool: "shell", arguments: '{"command":"ls"}' }] },
        { role: "tool_result", tool: "shell", content: '{"output":"main.py\\nREADME.md"}' },
      ],
    });
  });

  it("forces tool selection by updating toolConfig.functionCallingConfig", async () => {
    const { post, upstream } = setup(shellDecision);
    await post(geminiRequest());

    expect(upstream.calls).toHaveLength(1);
    const sent = upstream.calls[0]!.body as {
      toolConfig?: { functionCallingConfig?: { mode: string; allowedFunctionNames?: string[] } };
    };
    expect(sent.toolConfig?.functionCallingConfig).toEqual({
      mode: "ANY",
      allowedFunctionNames: ["shell"],
    });
  });

  it("answers directly without an upstream call when all arguments are resolved", async () => {
    const { post, upstream } = setup({
      tool: { choice: "set_lights" },
      needs_tool: { noul: 0.95 },
      "arg:0:room": { choice: "bedroom" },
      "arg:0:on": { noul: 0.99 },
    });
    const response = await post(
      geminiRequest({
        tools: [
          {
            functionDeclarations: [
              {
                name: "set_lights",
                description: "Turn lights on or off",
                parameters: {
                  type: "object",
                  properties: {
                    room: { type: "string", enum: ["kitchen", "bedroom"] },
                    on: { type: "boolean" },
                  },
                  required: ["room", "on"],
                },
              },
            ],
          },
        ],
      }),
    );

    expect(upstream.calls).toHaveLength(0);
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      candidates: Array<{ content: { parts: Array<{ functionCall?: { name: string; args: unknown } }> } }>;
    };
    expect(body.candidates[0]!.content.parts[0]!.functionCall).toEqual({
      name: "set_lights",
      args: { room: "bedroom", on: true },
    });
  });

  it("disables tools when Jev is confident no tool is needed", async () => {
    const { post, upstream } = setup({
      tool: { choice: NO_TOOL },
      needs_tool: { noul: 0.05 },
    });
    await post(geminiRequest());

    expect(upstream.calls).toHaveLength(1);
    const sent = upstream.calls[0]!.body as {
      toolConfig?: { functionCallingConfig?: { mode: string } };
    };
    expect(sent.toolConfig?.functionCallingConfig).toEqual({
      mode: "NONE",
    });
  });

  it("auto-detects Gemini wire format in /router/decide", async () => {
    const { app } = setup(shellDecision);
    const res = await app.request("/router/decide", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(geminiRequest()),
    });

    expect(res.status).toBe(200);
    const decision = (await res.json()) as { mode: string; tool?: string };
    expect(decision.mode).toBe("forced");
    expect(decision.tool).toBe("shell");
  });

  it("sends /v1beta paths upstream as they came, query included", async () => {
    const jev = fakeJev({ tool: { choice: "shell", confidence: 0.2 }, needs_tool: { noul: 0.9 } });
    const upstream = fakeUpstream();
    const app = createApp({
      config: testConfig({ upstreamBaseUrl: "https://generativelanguage.googleapis.com" }),
      askJev: jev.askJev,
      fetch: upstream.fetchImpl,
    });
    await app.request("/v1beta/models/gemini-2.5-pro:generateContent?key=abc", { method: "POST", body: JSON.stringify(geminiRequest()) });
    await app.request("/v1beta/models?pageSize=5");
    expect(upstream.calls.map((call) => call.url)).toEqual([
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=abc",
      "https://generativelanguage.googleapis.com/v1beta/models?pageSize=5",
    ]);
  });

  it("reads the model and the choice to stream from the path", async () => {
    const noArgs = { functionDeclarations: [{ name: "list_plans", description: "List saved plans.", parameters: { type: "object", properties: {} } }] };
    const canned = { tool: { choice: "list_plans" }, needs_tool: { noul: 0.9 } };
    const direct = async (path: string) => {
      const app = createApp({ config: testConfig(), askJev: fakeJev(canned).askJev, fetch: fakeUpstream().fetchImpl });
      const res = await app.request(path, { method: "POST", body: JSON.stringify(geminiRequest({ tools: [noArgs] })) });
      await settled();
      const feed = (await (await app.request("/dashboard/events")).json()) as { events: { model?: string }[] };
      return { type: res.headers.get("content-type"), text: await res.text(), model: feed.events[0]?.model };
    };

    const sse = await direct("/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse");
    expect(sse.type).toContain("text/event-stream");
    expect(JSON.parse(sse.text.replace(/^data: /, "")).candidates[0].content.parts[0].functionCall.name).toBe("list_plans");
    expect(sse.model).toBe("gemini-2.5-pro");

    const array = await direct("/v1beta/models/gemini-2.5-pro:streamGenerateContent");
    expect(array.type).toContain("application/json");
    expect(JSON.parse(array.text)[0].usageMetadata).toEqual({ promptTokenCount: 123, candidatesTokenCount: 0, totalTokenCount: 123 });

    expect((await direct("/v1beta/models/gemini-2.5-pro:generateContent")).type).toContain("application/json");
    const malformedOperation = await direct("/v1beta/models/gemini-2.5-pro:countTokens:streamGenerateContent");
    expect(Array.isArray(JSON.parse(malformedOperation.text))).toBe(false);
  });

  it("offers Google-run tools to Jev without forcing them, and respects allowedFunctionNames", async () => {
    const two = { functionDeclarations: [{ name: "shell", description: "Runs a shell command." }, { name: "read_file", description: "Reads a file." }] };
    const hosted = setup({ tool: { choice: "googleSearch" }, needs_tool: { noul: 0.9 } });
    const res = await hosted.post(geminiRequest({ tools: [two, { googleSearch: {} }] }));
    expect(res.headers.get("x-jev-gateway-reason")).toBe("hosted_tool_selected");

    const narrowed = setup({ tool: { choice: "read_file" }, needs_tool: { noul: 0.9 } });
    await narrowed.post(geminiRequest({ tools: [two], toolConfig: { functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["read_file"] } } }));
    const question = narrowed.jev.requests[0]!.questions.tool!;
    expect(question.type === "choice" && Object.keys(question.criteria)).toEqual(["read_file"]);
  });

  it("meters a streamed Gemini reply", async () => {
    const body = `data: ${JSON.stringify({ candidates: [], usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 20, cachedContentTokenCount: 64 } })}\n\n`;
    expect(await readUsage(new Response(body))).toMatchObject({ input: 100, output: 20, cached: 64 });
  });

  describe("Cloud Code and internal /v1internal routing", () => {
    const wrappedInternalRequest = (extra: Record<string, unknown> = {}) => ({
      project: "projects/test-project/locations/global",
      model: "gemini-3.8-flash-high",
      requestId: "req-123",
      request: geminiRequest(extra),
    });

    it("filters out thinking blocks (thought: true) from conversation turns", async () => {
      const { post, jev } = setup(shellDecision);
      const reqWithThought = wrappedInternalRequest({
        contents: [
          {
            role: "user",
            parts: [{ text: "what does main.py do?" }],
          },
          {
            role: "model",
            parts: [
              { text: "Thinking about the files in the directory...", thought: true },
              {
                functionCall: {
                  name: "hidden_thought_call",
                  args: { command: "do not include" },
                },
                thought: true,
              },
              {
                functionCall: {
                  name: "shell",
                  args: { command: "ls" },
                },
              },
            ],
          },
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  name: "shell",
                  response: { output: "main.py\nREADME.md" },
                },
              },
            ],
          },
        ],
      });
      await post(reqWithThought, "/v1internal:streamGenerateContent");

      const { state } = jev.requests[0]!;
      expect((state as Record<string, unknown>).conversation).toEqual([
        { role: "user", text: "what does main.py do?" },
        { role: "assistant", tool_calls: [{ tool: "shell", arguments: '{"command":"ls"}' }] },
        { role: "tool_result", tool: "shell", content: "{\"output\":\"main.py\\nREADME.md\"}" },
      ]);
    });

    it("records model from top-level or wrapped request in dashboard events", async () => {
      const { post, app } = setup(shellDecision);
      await post(wrappedInternalRequest(), "/v1internal:streamGenerateContent");
      await settled();

      const feed = (await (await app.request("/dashboard/events")).json()) as { events: { model?: string }[] };
      expect(feed.events[0]?.model).toBe("gemini-3.8-flash-high");
    });

    it("records a nested model when the envelope has no top-level model", async () => {
      const { post, app } = setup(shellDecision);
      const body = { ...wrappedInternalRequest(), model: undefined, request: geminiRequest({ model: "nested-gemini" }) };
      await post(body, "/v1internal:generateContent");
      await settled();

      const feed = (await (await app.request("/dashboard/events")).json()) as { events: { model?: string }[] };
      expect(feed.events[0]?.model).toBe("nested-gemini");
    });

    it("disables tools inside a wrapped request when Jev is confident no tool is needed", async () => {
      const { post, upstream } = setup({ tool: { choice: NO_TOOL }, needs_tool: { noul: 0.05 } });
      await post(wrappedInternalRequest(), "/v1internal:generateContent");

      const sent = upstream.calls[0]!.body as {
        request?: { toolConfig?: { functionCallingConfig?: { mode: string } } };
      };
      expect(sent.request?.toolConfig?.functionCallingConfig).toEqual({ mode: "NONE" });
    });

    it("passes a malformed internal envelope upstream without asking Jev", async () => {
      const { post, upstream, jev } = setup(shellDecision);
      const malformed = { ...geminiRequest(), request: [] };
      await post(malformed, "/v1internal:generateContent");

      expect(jev.requests).toHaveLength(0);
      expect(upstream.calls[0]?.body).toEqual(malformed);
    });

    it.each([400, 422])("replays the original wrapped body when the provider rejects a rewrite with %i", async (status) => {
      const jev = fakeJev(shellDecision);
      const sent: unknown[] = [];
      const fetchImpl = (async (_input: RequestInfo | URL, init?: RequestInit) => {
        const body = init?.body;
        const text = typeof body === "string" ? body : body instanceof Uint8Array ? Buffer.from(body).toString("utf8") : "";
        sent.push(JSON.parse(text));
        return sent.length === 1 ? Response.json({ error: "unsupported rewrite" }, { status }) : Response.json({ ok: true });
      }) as typeof fetch;
      const app = createApp({ config: testConfig(), askJev: jev.askJev, fetch: fetchImpl });
      const original = wrappedInternalRequest();
      const response = await app.request("/v1internal:generateContent", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(original),
      });

      expect(sent).toHaveLength(2);
      const rewritten = sent[0] as {
        request: { toolConfig?: { functionCallingConfig?: { mode: string; allowedFunctionNames?: string[] } } };
      };
      expect(rewritten.request.toolConfig?.functionCallingConfig).toEqual({
        mode: "ANY",
        allowedFunctionNames: ["shell"],
      });
      expect(sent[1]).toEqual(original);
      expect(response.headers.get("x-jev-gateway-reason")).toBe("upstream_rejected_forced");
    });

    it("translates Cloud Code internal wrapped request into Jev turns and tool declarations", async () => {
      const { post, jev } = setup(shellDecision);
      await post(wrappedInternalRequest(), "/v1internal:generateContent");

      const { state } = jev.requests[0]!;
      expect(state).toEqual({
        assistant_instructions: "You are a helpful coding assistant.",
        conversation: [
          { role: "user", text: "what does main.py do?" },
          { role: "assistant", tool_calls: [{ tool: "shell", arguments: '{"command":"ls"}' }] },
        { role: "tool_result", tool: "shell", content: '{"output":"main.py\\nREADME.md"}' },
        ],
      });
    });

    it("forces tool selection in wrapped request.toolConfig", async () => {
      const { post, upstream } = setup(shellDecision);
      await post(wrappedInternalRequest(), "/v1internal:streamGenerateContent");

      expect(upstream.calls).toHaveLength(1);
      const sent = upstream.calls[0]!.body as {
        request?: { toolConfig?: { functionCallingConfig?: { mode: string; allowedFunctionNames?: string[] } } };
      };
      expect(sent.request?.toolConfig?.functionCallingConfig).toEqual({
        mode: "ANY",
        allowedFunctionNames: ["shell"],
      });
    });

    it("returns direct answers wrapped in response object for internal requests", async () => {
      const { post, upstream } = setup({
        tool: { choice: "set_lights" },
        needs_tool: { noul: 0.95 },
        "arg:0:room": { choice: "bedroom" },
        "arg:0:on": { noul: 0.99 },
      });
      const response = await post(
        wrappedInternalRequest({
          tools: [
            {
              functionDeclarations: [
                {
                  name: "set_lights",
                  description: "Turn lights on or off",
                  parameters: {
                    type: "object",
                    properties: {
                      room: { type: "string", enum: ["kitchen", "bedroom"] },
                      on: { type: "boolean" },
                    },
                    required: ["room", "on"],
                  },
                },
              ],
            },
          ],
        }),
        "/v1internal:generateContent",
      );

      expect(upstream.calls).toHaveLength(0);
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        response: {
          candidates: Array<{ content: { parts: Array<{ functionCall?: { name: string; args: unknown } }> } }>;
        };
      };
      expect(body.response.candidates[0]!.content.parts[0]!.functionCall).toEqual({
        name: "set_lights",
        args: { room: "bedroom", on: true },
      });
    });

    it("returns a JSON array for internal stream generation unless alt=sse is requested", async () => {
      const canned = {
        tool: { choice: "list_plans" },
        needs_tool: { noul: 0.9 },
      };
      const noArgs = { functionDeclarations: [{ name: "list_plans", parameters: { type: "object", properties: {} } }] };
      const body = wrappedInternalRequest({ tools: [noArgs] });
      const direct = async (path: string) => {
        const { post } = setup(canned);
        const response = await post(body, path);
        return { response, text: await response.text() };
      };

      const array = await direct("/v1internal:streamGenerateContent");
      expect(array.response.headers.get("content-type")).toContain("application/json");
      expect(Array.isArray(JSON.parse(array.text))).toBe(true);

      const sse = await direct("/v1internal:streamGenerateContent?alt=sse");
      expect(sse.response.headers.get("content-type")).toContain("text/event-stream");
      expect(sse.text).toContain("data: ");
    });

    it("streams direct answers as SSE on /v1internal:streamGenerateContent", async () => {
      const { post, upstream } = setup({
        tool: { choice: "set_lights" },
        needs_tool: { noul: 0.95 },
        "arg:0:room": { choice: "bedroom" },
        "arg:0:on": { noul: 0.99 },
      });
      const response = await post(
        wrappedInternalRequest({
          tools: [
            {
              functionDeclarations: [
                {
                  name: "set_lights",
                  description: "Turn lights on or off",
                  parameters: {
                    type: "object",
                    properties: {
                      room: { type: "string", enum: ["kitchen", "bedroom"] },
                      on: { type: "boolean" },
                    },
                    required: ["room", "on"],
                  },
                },
              ],
            },
          ],
        }),
        "/v1internal:streamGenerateContent?alt=sse",
      );
      expect(upstream.calls).toHaveLength(0);
      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toContain("text/event-stream");
      const text = await response.text();
      const chunk = JSON.parse(text.replace(/^data: /, "").trim());
      expect(chunk.response.candidates[0].content.parts[0].functionCall.name).toBe("set_lights");
    });

    it("proxies management calls untouched with their credentials and headers", async () => {
      const jev = fakeJev(shellDecision);
      const calls: { url: string; headers: Headers; body: unknown }[] = [];
      const fetchImpl = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const body = init?.body;
        calls.push({
          url: String(input),
          headers: new Headers(init?.headers),
          body: body ? await new Response(body).json() : undefined,
        });
        return Response.json({ ok: true });
      }) as typeof fetch;
      const app = createApp({
        config: testConfig({ upstreamBaseUrl: "https://daily-cloudcode-pa.googleapis.com" }),
        askJev: jev.askJev,
        fetch: fetchImpl,
      });
      const headers = {
        authorization: "Bearer client-oauth-token",
        "content-type": "application/json",
        "x-goog-api-key": "client-api-key",
        "x-goog-api-client": "test-client/1",
      };

      await app.request("/v1internal:loadCodeAssist", { method: "POST", headers, body: JSON.stringify({ project: "proj-1" }) });
      await app.request("/v1internal:fetchAvailableModels", { method: "POST", headers, body: "{}" });
      await app.request("/v1internal/models/gemini-3.8-flash-high", {
        method: "POST",
        headers,
        body: JSON.stringify(geminiRequest()),
      });
      await app.request("/v1beta/projects/test-project/locations/global", {
        method: "POST",
        headers,
        body: JSON.stringify(geminiRequest()),
      });
      await app.request("/v1beta1/projects/test-project/locations/global", {
        method: "POST",
        headers,
        body: JSON.stringify(geminiRequest()),
      });

      expect(calls.map((c) => c.url)).toEqual([
        "https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
        "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
        "https://daily-cloudcode-pa.googleapis.com/v1internal/models/gemini-3.8-flash-high",
        "https://daily-cloudcode-pa.googleapis.com/v1beta/projects/test-project/locations/global",
        "https://daily-cloudcode-pa.googleapis.com/v1beta1/projects/test-project/locations/global",
      ]);
      for (const call of calls) {
        expect(call.headers.get("authorization")).toBe(headers.authorization);
        expect(call.headers.get("x-goog-api-key")).toBe(headers["x-goog-api-key"]);
        expect(call.headers.get("x-goog-api-client")).toBe(headers["x-goog-api-client"]);
        expect(call.headers.get("content-type")).toBe(headers["content-type"]);
      }
      expect(jev.requests).toHaveLength(0);
      expect(calls[2]?.body).toEqual(geminiRequest());
      expect(calls[3]?.body).toEqual(geminiRequest());
      expect(calls[4]?.body).toEqual(geminiRequest());
    });

    it("auto-detects wrapped internal request in /router/decide", async () => {
      const { app } = setup(shellDecision);
      const res = await app.request("/router/decide", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(wrappedInternalRequest()),
      });

      expect(res.status).toBe(200);
      const decision = (await res.json()) as { mode: string; tool?: string };
      expect(decision.mode).toBe("forced");
      expect(decision.tool).toBe("shell");
    });
  });
});
