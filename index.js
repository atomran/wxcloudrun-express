const path = require("path");
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const fetch = require("node-fetch");

const app = express();
const port = process.env.PORT || 80;
const model = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const maxBodySize = process.env.MAX_BODY_SIZE || "18mb";

app.use(express.urlencoded({ extended: false, limit: maxBodySize }));
app.use(express.json({ limit: maxBodySize }));
app.use(cors());
app.use(morgan("tiny"));

app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.get("/health", (_req, res) => {
  res.send({
    code: 0,
    data: {
      ok: true,
      service: "饮食训练日志识别服务",
      model,
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
  if (!process.env.OPENAI_API_KEY) {
    throwHttp(503, "Missing OPENAI_API_KEY");
  }

  if (!body?.image?.dataUrl || !String(body.image.dataUrl).startsWith("data:image/")) {
    throwHttp(400, "image.dataUrl is required");
  }

  const prompt = buildPrompt(body.type, body.date, body.hint, body.image.label);
  const apiResponse = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
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
              image_url: body.image.dataUrl,
            },
          ],
        },
      ],
    }),
  });

  const payload = await apiResponse.json();
  if (!apiResponse.ok) {
    throwHttp(502, payload?.error?.message || `OpenAI API HTTP ${apiResponse.status}`);
  }

  const text = extractOutputText(payload);
  const parsed = parseJsonFromText(text);
  return normalizeResult(parsed, body.type, body.date, text);
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

function extractOutputText(payload) {
  if (typeof payload.output_text === "string") return payload.output_text;
  const pieces = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === "string") pieces.push(content.text);
    }
  }
  return pieces.join("\n").trim();
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
