const path = require("path");
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const fetch = require("node-fetch");

const app = express();
const port = process.env.PORT || 80;
const maxBodySize = process.env.MAX_BODY_SIZE || "24mb";

app.use(express.urlencoded({ extended: false, limit: maxBodySize }));
app.use(express.json({ limit: maxBodySize }));
app.use(cors());
app.use(morgan("tiny"));

app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.get("/health", (_req, res) => {
  const provider = getProvider();
  res.send({
    code: 0,
    data: {
      ok: true,
      service: "饮食训练日志识别服务",
      provider,
      model: getModel(provider),
      timeoutMs: getAiTimeoutMs(),
      ready: hasProviderKey(provider),
      hasDashScopeKey: Boolean(getDashScopeApiKey()),
      hasGeminiKey: Boolean(getGeminiApiKey()),
      hasOpenAIKey: Boolean(process.env.OPENAI_API_KEY),
    },
  });
});

app.post("/api/count", (req, res) => {
  res.send({
    code: 0,
    data: {
      action: req.body?.action || "noop",
      message: "CloudBase Express service is alive",
    },
  });
});

app.get("/api/count", (_req, res) => {
  res.send({
    code: 0,
    data: {
      message: "CloudBase Express service is alive",
    },
  });
});

app.get("/api/wx_openid", (req, res) => {
  if (req.headers["x-wx-source"]) {
    res.send(req.headers["x-wx-openid"]);
    return;
  }
  res.status(400).send({
    code: -1,
    error: "Missing WeChat cloud container headers",
  });
});

app.post("/api/recognize-image", async (req, res, next) => {
  try {
    const result = await recognizeImage(req.body || {});
    res.send({
      code: 0,
      data: result,
    });
  } catch (error) {
    next(error);
  }
});

app.post("/api/training-advice", async (req, res, next) => {
  try {
    const result = await generateTrainingAdvice(req.body || {});
    res.send({
      code: 0,
      data: result,
    });
  } catch (error) {
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  const status = error.statusCode || 500;
  res.status(status).send({
    code: -1,
    error: error.message || "Recognition failed",
  });
});

app.listen(port, () => {
  console.log("启动成功", port);
});

async function recognizeImage(body) {
  const provider = getProvider();
  if (!hasProviderKey(provider)) {
    throwHttp(503, `Missing API key for provider: ${provider}`);
  }

  if (!body?.image?.dataUrl || !String(body.image.dataUrl).startsWith("data:image/")) {
    throwHttp(400, "image.dataUrl is required");
  }

  const prompt = buildPrompt(body.type, body.date, body.hint, body.image.label);
  const text = await callVisionModel(provider, prompt, body.image.dataUrl);
  const parsed = parseJsonFromText(text);
  return normalizeResult(parsed, body.type, body.date, text);
}

async function callVisionModel(provider, prompt, dataUrl) {
  if (provider === "qwen") return callQwenImage(prompt, dataUrl);
  if (provider === "gemini") return callGeminiImage(prompt, dataUrl);
  return callOpenAIImage(prompt, dataUrl);
}

async function callQwenImage(prompt, dataUrl) {
  const apiKey = getDashScopeApiKey();
  if (!apiKey) throwHttp(503, "Missing DASHSCOPE_API_KEY");

  const apiResponse = await fetchWithHttpError("DashScope Qwen-VL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    timeout: getAiTimeoutMs(),
    body: JSON.stringify({
      model: getModel("qwen"),
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: prompt,
            },
            {
              type: "image_url",
              image_url: {
                url: dataUrl,
              },
            },
          ],
        },
      ],
      temperature: 0.1,
    }),
  });

  const payload = await safeJson(apiResponse);
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `DashScope API HTTP ${apiResponse.status}`);
  }

  return extractOpenAIChatText(payload, "DashScope Qwen-VL");
}

async function callGeminiImage(prompt, dataUrl) {
  const apiKey = getGeminiApiKey();
  if (!apiKey) throwHttp(503, "Missing GEMINI_API_KEY");

  const inlineImage = parseDataUrl(dataUrl);
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(getModel("gemini"))}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const apiResponse = await fetchWithHttpError("Gemini", endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    timeout: getAiTimeoutMs(),
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts: [
            {
              text: prompt,
            },
            {
              inline_data: {
                mime_type: inlineImage.mimeType,
                data: inlineImage.base64,
              },
            },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.1,
        response_mime_type: "application/json",
      },
    }),
  });

  const payload = await safeJson(apiResponse);
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `Gemini API HTTP ${apiResponse.status}`);
  }

  return extractGeminiOutputText(payload);
}

async function callOpenAIImage(prompt, dataUrl) {
  if (!process.env.OPENAI_API_KEY) {
    throwHttp(503, "Missing OPENAI_API_KEY");
  }

  const apiResponse = await fetchWithHttpError("OpenAI", "https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    timeout: getAiTimeoutMs(),
    body: JSON.stringify({
      model: getModel("openai"),
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: prompt,
            },
            {
              type: "input_image",
              image_url: dataUrl,
            },
          ],
        },
      ],
    }),
  });

  const payload = await safeJson(apiResponse);
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `OpenAI API HTTP ${apiResponse.status}`);
  }

  return extractOpenAIOutputText(payload);
}

async function generateTrainingAdvice(body) {
  const provider = getProvider();
  if (!hasProviderKey(provider)) {
    throwHttp(503, `Missing API key for provider: ${provider}`);
  }

  const records = Array.isArray(body.records) ? body.records.slice(0, 14) : [];
  const current = body.current || {};
  const goals = body.goals || {};
  const prompt = buildTrainingAdvicePrompt(records, current, body.estimatedEnergy, goals);
  const text = await callTextModel(provider, prompt);
  const parsed = parseJsonFromText(text);

  return normalizeTrainingAdvice(parsed, text);
}

async function callTextModel(provider, prompt) {
  if (provider === "qwen") return callQwenText(prompt);
  if (provider === "openai") return callOpenAIText(prompt);
  throwHttp(400, `Training advice is not configured for provider: ${provider}`);
}

async function callQwenText(prompt) {
  const apiKey = getDashScopeApiKey();
  if (!apiKey) throwHttp(503, "Missing DASHSCOPE_API_KEY");

  const apiResponse = await fetchWithHttpError("DashScope Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    timeout: getAiTimeoutMs(),
    body: JSON.stringify({
      model: getTextModel("qwen"),
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.25,
      response_format: {
        type: "json_object",
      },
    }),
  });

  const payload = await safeJson(apiResponse);
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `DashScope API HTTP ${apiResponse.status}`);
  }

  return extractOpenAIChatText(payload, "DashScope Qwen");
}

async function callOpenAIText(prompt) {
  if (!process.env.OPENAI_API_KEY) {
    throwHttp(503, "Missing OPENAI_API_KEY");
  }

  const apiResponse = await fetchWithHttpError("OpenAI", "https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    timeout: getAiTimeoutMs(),
    body: JSON.stringify({
      model: getTextModel("openai"),
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: prompt,
            },
          ],
        },
      ],
    }),
  });

  const payload = await safeJson(apiResponse);
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `OpenAI API HTTP ${apiResponse.status}`);
  }

  return extractOpenAIOutputText(payload);
}

function getProvider() {
  const explicit = String(process.env.AI_PROVIDER || "").toLowerCase();
  if (explicit === "qwen" || explicit === "dashscope" || explicit === "aliyun") return "qwen";
  if (explicit === "gemini" || explicit === "openai") return explicit;
  if (getDashScopeApiKey()) return "qwen";
  if (getGeminiApiKey()) return "gemini";
  return "openai";
}

function getModel(provider) {
  if (provider === "qwen") return process.env.QWEN_MODEL || process.env.DASHSCOPE_MODEL || "qwen-vl-plus";
  if (provider === "gemini") return process.env.GEMINI_MODEL || "gemini-2.5-flash";
  return process.env.OPENAI_MODEL || "gpt-4.1-mini";
}

function getTextModel(provider) {
  if (provider === "qwen") return process.env.QWEN_TEXT_MODEL || "qwen-plus";
  return process.env.OPENAI_TEXT_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini";
}

function getDashScopeApiKey() {
  return process.env.DASHSCOPE_API_KEY || process.env.DASHSCOPE_API_KEY_ID || "";
}

function getGeminiApiKey() {
  return process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "";
}

function hasProviderKey(provider) {
  if (provider === "qwen") return Boolean(getDashScopeApiKey());
  if (provider === "gemini") return Boolean(getGeminiApiKey());
  return Boolean(process.env.OPENAI_API_KEY);
}

function getAiTimeoutMs() {
  const parsed = Number(process.env.AI_TIMEOUT_MS);
  if (Number.isFinite(parsed) && parsed >= 5000) return parsed;
  return 25000;
}

function parseDataUrl(dataUrl) {
  const match = String(dataUrl || "").match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([\s\S]+)$/);
  if (!match) throwHttp(400, "image.dataUrl must be a base64 data URL");
  return {
    mimeType: match[1],
    base64: match[2],
  };
}

function buildPrompt(type, date, hint, label) {
  return [
    "你是个人身体记录应用的视觉识别模块。请从图片中提取可确认的数据，返回严格 JSON，不要返回 Markdown。",
    `图片类型: ${type || "unknown"}`,
    `默认日期: ${date || ""}`,
    `文件名: ${label || ""}`,
    `用户补充: ${hint || ""}`,
    "如果看不清或无法判断，字段用 null 或空字符串，不要编造。",
    "饮食照片请估算食物名称、热量、蛋白、碳水、脂肪，并在 warnings 里提醒这是估算。",
    "小米体脂秤截图请尽量提取体重、体脂率、BMI、肌肉量、基础代谢、内脏脂肪、身体年龄。",
    "训练截图请提取训练类型、时长、强度、计划或完成内容。",
    "攀岩截图请提取场地、难度、尝试次数、完攀数量和技术备注。",
    "体型照片只需要形成照片备注，不要评价外貌。",
    "JSON schema:",
    JSON.stringify({
      type: "weight | diet | training | climbing | body",
      date: "YYYY-MM-DD",
      confidence: 0.0,
      summary: "short human readable summary",
      warnings: ["string"],
      rawText: "visible text or brief visual evidence",
      draft: {
        date: "YYYY-MM-DD",
        weight: null,
        sleepHours: null,
        energy: null,
        bodyMetrics: {
          bodyFatPercent: null,
          bmi: null,
          muscleKg: null,
          basalMetabolism: null,
          visceralFat: null,
          bodyAge: null,
        },
        diet: {
          calories: null,
          protein: null,
          carbs: null,
          fat: null,
          meals: "",
        },
        training: {
          type: "strength | climbing | rest",
          durationMinutes: null,
          intensity: null,
          plan: "",
          completed: "",
        },
        climbing: {
          didClimb: false,
          location: "",
          grade: "",
          attempts: null,
          sends: null,
          notes: "",
        },
        notes: "",
      },
    }),
  ].join("\n");
}

function buildTrainingAdvicePrompt(records, current, estimatedEnergy, goals) {
  return [
    "你是个人训练日志应用里的训练建议模块。请根据用户最近记录、当前训练内容、体重和估算能量消耗，生成下一周训练建议和渐进计划。",
    "要求：只返回严格 JSON，不要 Markdown。建议要保守、可执行、避免医疗诊断；如果数据不足要明确说明。",
    "重点考虑：训练频率、训练时长、强度、力量训练、攀岩、恢复、体重变化、饮食热量和蛋白质。",
    "如果用户设置了每日热量、蛋白、目标体重或每周训练分钟数，请围绕这些目标给出可执行建议。",
    "如果出现明显疲劳、训练量过高、睡眠差或连续高强度，要建议降载或休息。",
    "输入数据：",
    JSON.stringify({
      current,
      goals,
      estimatedEnergy,
      recentRecords: records,
    }),
    "JSON schema:",
    JSON.stringify({
      summary: "one sentence",
      weeklyAdvice: ["string"],
      progressionPlan: [
        {
          day: "周一",
          focus: "strength | climbing | recovery | cardio | rest",
          session: "specific plan",
          targetMinutes: 60,
          intensity: 3,
          progression: "how to progress",
        },
      ],
      estimatedEnergyComment: "string",
      recoveryFlags: ["string"],
      nutritionNotes: ["string"],
      nextCheckIn: "what to review next week",
      disclaimer: "short caution",
    }),
  ].join("\n");
}

function normalizeTrainingAdvice(result, rawText) {
  return {
    ok: true,
    summary: result.summary || "",
    weeklyAdvice: Array.isArray(result.weeklyAdvice) ? result.weeklyAdvice : [],
    progressionPlan: Array.isArray(result.progressionPlan) ? result.progressionPlan : [],
    estimatedEnergyComment: result.estimatedEnergyComment || "",
    recoveryFlags: Array.isArray(result.recoveryFlags) ? result.recoveryFlags : [],
    nutritionNotes: Array.isArray(result.nutritionNotes) ? result.nutritionNotes : [],
    nextCheckIn: result.nextCheckIn || "",
    disclaimer: result.disclaimer || "建议仅用于训练记录和计划参考，如有不适请降低强度或咨询专业人士。",
    rawText,
  };
}

function normalizeResult(result, type, date, rawText) {
  const draft = result.draft || {};
  draft.date = draft.date || result.date || date || new Date().toISOString().slice(0, 10);

  return {
    ok: true,
    type: result.type || type,
    confidence: clamp(Number(result.confidence), 0, 1, 0.72),
    summary: result.summary || "",
    warnings: Array.isArray(result.warnings) ? result.warnings : [],
    rawText: result.rawText || rawText || "",
    draft,
  };
}

function extractOpenAIOutputText(payload) {
  if (typeof payload.output_text === "string") return payload.output_text;
  const pieces = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === "string") pieces.push(content.text);
    }
  }
  return pieces.join("\n").trim();
}

function extractOpenAIChatText(payload, provider) {
  const content = payload?.choices?.[0]?.message?.content;
  if (typeof content === "string" && content.trim()) return content.trim();
  if (Array.isArray(content)) {
    const pieces = content
      .map((part) => typeof part?.text === "string" ? part.text : "")
      .filter(Boolean);
    if (pieces.length) return pieces.join("\n").trim();
  }
  throwHttp(502, `${provider} did not return text`);
}

function extractGeminiOutputText(payload) {
  if (typeof payload.output_text === "string") return payload.output_text;

  const pieces = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === "string") pieces.push(content.text);
    }
  }

  for (const candidate of payload.candidates || []) {
    for (const part of candidate.content?.parts || []) {
      if (typeof part.text === "string") pieces.push(part.text);
    }
  }

  const text = pieces.join("\n").trim();
  if (!text) throwHttp(502, "Gemini did not return text");
  return text;
}

async function safeJson(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return {
      error: {
        message: text || `HTTP ${response.status}`,
      },
    };
  }
}

async function fetchWithHttpError(provider, url, options) {
  try {
    return await fetch(url, options);
  } catch (error) {
    const timeoutMs = options?.timeout || getAiTimeoutMs();
    if (String(error?.message || "").toLowerCase().includes("timeout")) {
      throwHttp(502, `${provider} request timed out after ${timeoutMs}ms`);
    }
    throwHttp(502, `${provider} request failed: ${error.message || "network error"}`);
  }
}

function parseJsonFromText(text) {
  const trimmed = String(text || "").trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) throwHttp(502, "Model did not return JSON");
    return JSON.parse(match[0]);
  }
}

function clamp(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function throwHttp(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  throw error;
}
