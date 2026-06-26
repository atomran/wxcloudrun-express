const path = require("path");
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const fetch = require("node-fetch");
const multer = require("multer");
const mysql = require("mysql2/promise");

const app = express();
const port = process.env.PORT || 80;
const maxBodySize = process.env.MAX_BODY_SIZE || "24mb";
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: parseSizeToBytes(maxBodySize, 24 * 1024 * 1024),
  },
});

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
      hasMysql: Boolean(process.env.MYSQL_ADDRESS && process.env.MYSQL_USERNAME),
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

app.get("/api/state", async (req, res, next) => {
  try {
    const result = await loadUserState(req);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.post("/api/state", async (req, res, next) => {
  try {
    const result = await saveUserState(req, req.body || {});
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.delete("/api/state", async (req, res, next) => {
  try {
    const result = await deleteUserState(req);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.post("/api/record", async (req, res, next) => {
  try {
    const result = await saveUserRecord(req, req.body?.record);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.delete("/api/record", async (req, res, next) => {
  try {
    const result = await deleteUserRecord(req, req.body?.id);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.post("/api/photo", async (req, res, next) => {
  try {
    const result = await saveUserPhoto(req, req.body?.photo);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
});

app.delete("/api/photo", async (req, res, next) => {
  try {
    const result = await deleteUserPhoto(req, req.body?.id);
    res.send({ code: 0, data: result });
  } catch (error) {
    next(error);
  }
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

app.post("/api/recognize-image-url", async (req, res, next) => {
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

app.post("/api/recognize-upload", upload.single("image"), async (req, res, next) => {
  try {
    const result = await recognizeUpload(req.body || {}, req.file);
    res.send({
      code: 0,
      data: result,
    });
  } catch (error) {
    next(error);
  }
});

app.post("/api/estimate-diet-text", async (req, res, next) => {
  try {
    const result = await estimateDietText(req.body || {});
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

let mysqlPool = null;
let stateTableReadyPromise = null;

async function loadUserState(req) {
  const openid = requireOpenid(req);
  await ensureStateTable();
  const pool = getMysqlPool();
  const [rows] = await pool.execute(
    "SELECT openid, display_name, profile_json, goals_json, records_json, photos_json, draft_form_json, updated_at FROM body_log_user_state WHERE openid = ? LIMIT 1",
    [openid]
  );
  const row = rows && rows[0];
  if (!row) {
    return {
      ok: true,
      exists: false,
      state: null,
    };
  }

  const [recordRows] = await pool.execute(
    "SELECT record_json FROM body_log_records WHERE openid = ? ORDER BY record_date DESC, updated_at DESC",
    [openid]
  );
  const [photoRows] = await pool.execute(
    "SELECT photo_json FROM body_log_photos WHERE openid = ? ORDER BY photo_date DESC, updated_at DESC",
    [openid]
  );
  const profile = parseJsonField(row.profile_json, {});
  profile.openidText = profile.displayName || profile.openidText || "微信用户";
  const records = recordRows.length
    ? recordRows.map((item) => parseJsonField(item.record_json, null)).filter(Boolean)
    : parseJsonField(row.records_json, []);
  const photos = photoRows.length
    ? photoRows.map((item) => parseJsonField(item.photo_json, null)).filter(Boolean)
    : parseJsonField(row.photos_json, []);
  return {
    ok: true,
    exists: true,
    state: {
      version: 1,
      records,
      photos,
      goals: parseJsonField(row.goals_json, {}),
      profile,
      draftForm: parseJsonField(row.draft_form_json, null),
      updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : new Date().toISOString(),
    },
  };
}

async function saveUserState(req, body) {
  const openid = requireOpenid(req);
  await ensureStateTable();
  const pool = getMysqlPool();

  const profile = sanitizeJsonObject(body.profile);
  profile.openid = openid;
  profile.openidText = profile.displayName || "微信用户";
  const goals = sanitizeJsonObject(body.goals);
  const records = Array.isArray(body.records) ? body.records : [];
  const photos = Array.isArray(body.photos) ? body.photos : [];
  const draftForm = body.draftForm && typeof body.draftForm === "object" ? body.draftForm : null;
  const displayName = String(profile.displayName || "").trim().slice(0, 64);

  await pool.execute(
    [
      "INSERT INTO body_log_user_state",
      "(openid, display_name, profile_json, goals_json, records_json, photos_json, draft_form_json)",
      "VALUES (?, ?, ?, ?, ?, ?, ?)",
      "ON DUPLICATE KEY UPDATE",
      "display_name = VALUES(display_name),",
      "profile_json = VALUES(profile_json),",
      "goals_json = VALUES(goals_json),",
      "records_json = VALUES(records_json),",
      "photos_json = VALUES(photos_json),",
      "draft_form_json = VALUES(draft_form_json),",
      "updated_at = CURRENT_TIMESTAMP",
    ].join(" "),
    [
      openid,
      displayName,
      JSON.stringify(profile),
      JSON.stringify(goals),
      JSON.stringify(records),
      JSON.stringify(photos),
      draftForm ? JSON.stringify(draftForm) : null,
    ]
  );

  await upsertRecords(openid, records);
  await upsertPhotos(openid, photos);

  return {
    ok: true,
    updatedAt: new Date().toISOString(),
  };
}

async function saveUserRecord(req, record) {
  const openid = requireOpenid(req);
  if (!record || typeof record !== "object" || !record.id) {
    throwHttp(400, "record.id is required");
  }
  await ensureStateTable();
  await ensureUserStateShell(openid);
  await upsertRecords(openid, [record]);
  return { ok: true, id: record.id, updatedAt: new Date().toISOString() };
}

async function deleteUserRecord(req, id) {
  const openid = requireOpenid(req);
  if (!id) throwHttp(400, "record id is required");
  await ensureStateTable();
  await getMysqlPool().execute(
    "DELETE FROM body_log_records WHERE openid = ? AND record_id = ?",
    [openid, String(id)]
  );
  return { ok: true, id };
}

async function saveUserPhoto(req, photo) {
  const openid = requireOpenid(req);
  if (!photo || typeof photo !== "object" || !photo.id) {
    throwHttp(400, "photo.id is required");
  }
  await ensureStateTable();
  await ensureUserStateShell(openid);
  await upsertPhotos(openid, [photo]);
  return { ok: true, id: photo.id, updatedAt: new Date().toISOString() };
}

async function deleteUserPhoto(req, id) {
  const openid = requireOpenid(req);
  if (!id) throwHttp(400, "photo id is required");
  await ensureStateTable();
  await getMysqlPool().execute(
    "DELETE FROM body_log_photos WHERE openid = ? AND photo_id = ?",
    [openid, String(id)]
  );
  return { ok: true, id };
}

async function deleteUserState(req) {
  const openid = requireOpenid(req);
  await ensureStateTable();
  const pool = getMysqlPool();
  await pool.execute("DELETE FROM body_log_records WHERE openid = ?", [openid]);
  await pool.execute("DELETE FROM body_log_photos WHERE openid = ?", [openid]);
  await pool.execute("DELETE FROM body_log_user_state WHERE openid = ?", [openid]);
  return { ok: true };
}

async function upsertRecords(openid, records) {
  const items = Array.isArray(records) ? records.filter((item) => item && item.id) : [];
  for (const record of items) {
    await getMysqlPool().execute(
      [
        "INSERT INTO body_log_records",
        "(openid, record_id, record_date, record_json)",
        "VALUES (?, ?, ?, ?)",
        "ON DUPLICATE KEY UPDATE",
        "record_date = VALUES(record_date),",
        "record_json = VALUES(record_json),",
        "updated_at = CURRENT_TIMESTAMP",
      ].join(" "),
      [
        openid,
        String(record.id),
        normalizeDateForSql(record.date),
        JSON.stringify(record),
      ]
    );
  }
}

async function ensureUserStateShell(openid) {
  await getMysqlPool().execute(
    [
      "INSERT INTO body_log_user_state",
      "(openid, display_name, profile_json, goals_json, records_json, photos_json, draft_form_json)",
      "VALUES (?, '', '{}', '{}', '[]', '[]', NULL)",
      "ON DUPLICATE KEY UPDATE openid = openid",
    ].join(" "),
    [openid]
  );
}

async function upsertPhotos(openid, photos) {
  const items = Array.isArray(photos) ? photos.filter((item) => item && item.id) : [];
  for (const photo of items) {
    await getMysqlPool().execute(
      [
        "INSERT INTO body_log_photos",
        "(openid, photo_id, photo_date, photo_json)",
        "VALUES (?, ?, ?, ?)",
        "ON DUPLICATE KEY UPDATE",
        "photo_date = VALUES(photo_date),",
        "photo_json = VALUES(photo_json),",
        "updated_at = CURRENT_TIMESTAMP",
      ].join(" "),
      [
        openid,
        String(photo.id),
        normalizeDateForSql(photo.date),
        JSON.stringify(photo),
      ]
    );
  }
}

function requireOpenid(req) {
  const fromHeader = req.headers["x-wx-openid"];
  const fromWx = Boolean(req.headers["x-wx-source"]);
  if (fromHeader && fromWx) return String(fromHeader);
  if (process.env.ALLOW_DEV_OPENID === "1" && fromHeader) return String(fromHeader);
  if (process.env.ALLOW_DEV_OPENID === "1" && req.body && req.body.openid) return String(req.body.openid);
  throwHttp(401, "Missing WeChat openid. Please call through wx.cloud.callContainer.");
}

function getMysqlPool() {
  if (mysqlPool) return mysqlPool;
  const username = process.env.MYSQL_USERNAME;
  const password = process.env.MYSQL_PASSWORD;
  const address = process.env.MYSQL_ADDRESS || "";
  const [host, portText] = address.split(":");
  if (!username || !password || !host) {
    throwHttp(503, "Missing MySQL environment variables");
  }

  mysqlPool = mysql.createPool({
    host,
    port: Number(portText) || 3306,
    user: username,
    password,
    database: process.env.MYSQL_DATABASE || "nodejs_demo",
    waitForConnections: true,
    connectionLimit: 4,
    charset: "utf8mb4",
  });
  return mysqlPool;
}

async function ensureStateTable() {
  if (!stateTableReadyPromise) {
    stateTableReadyPromise = (async () => {
      const pool = getMysqlPool();
      await pool.execute(`
        CREATE TABLE IF NOT EXISTS body_log_user_state (
          openid VARCHAR(128) NOT NULL PRIMARY KEY,
          display_name VARCHAR(128) DEFAULT '',
          profile_json MEDIUMTEXT NOT NULL,
          goals_json MEDIUMTEXT NOT NULL,
          records_json LONGTEXT NOT NULL,
          photos_json LONGTEXT NOT NULL,
          draft_form_json MEDIUMTEXT NULL,
          created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          KEY idx_updated_at (updated_at)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
      `);
      await pool.execute(`
        CREATE TABLE IF NOT EXISTS body_log_records (
          openid VARCHAR(128) NOT NULL,
          record_id VARCHAR(80) NOT NULL,
          record_date DATE NULL,
          record_json LONGTEXT NOT NULL,
          created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (openid, record_id),
          KEY idx_openid_date (openid, record_date)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
      `);
      await pool.execute(`
        CREATE TABLE IF NOT EXISTS body_log_photos (
          openid VARCHAR(128) NOT NULL,
          photo_id VARCHAR(80) NOT NULL,
          photo_date DATE NULL,
          photo_json MEDIUMTEXT NOT NULL,
          created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (openid, photo_id),
          KEY idx_openid_date (openid, photo_date)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
      `);
    })().catch((error) => {
      stateTableReadyPromise = null;
      throw error;
    });
  }
  await stateTableReadyPromise;
}

function normalizeDateForSql(value) {
  const text = String(value || "").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function parseJsonField(value, fallback) {
  if (value === null || value === undefined || value === "") return fallback;
  try {
    return JSON.parse(Buffer.isBuffer(value) ? value.toString("utf8") : String(value));
  } catch (_error) {
    return fallback;
  }
}

function sanitizeJsonObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

async function recognizeImage(body) {
  const provider = getProvider();
  if (!hasProviderKey(provider)) {
    throwHttp(503, `Missing API key for provider: ${provider}`);
  }

  const imageSource = body?.image?.dataUrl || body?.image?.url;
  if (!isSupportedImageSource(imageSource)) {
    throwHttp(400, "image.dataUrl or image.url is required");
  }

  const prompt = buildPrompt(body.type, body.date, body.hint, body.image.label);
  const text = await callVisionModel(provider, prompt, imageSource);
  const parsed = parseJsonFromText(text);
  return normalizeResult(parsed, body.type, body.date, text);
}

async function recognizeUpload(body, file) {
  if (!file || !file.buffer || !file.buffer.length) {
    throwHttp(400, "image file is required");
  }

  if (!String(file.mimetype || "").startsWith("image/")) {
    throwHttp(400, "uploaded file must be an image");
  }

  return recognizeImage({
    type: body.type,
    date: body.date,
    hint: body.hint,
    image: {
      label: body.label || file.originalname || "upload.jpg",
      dataUrl: `data:${file.mimetype};base64,${file.buffer.toString("base64")}`,
      width: Number(body.width) || null,
      height: Number(body.height) || null,
      uploadSize: file.size,
    },
  });
}

async function estimateDietText(body) {
  const text = String(body.text || "").trim();
  if (!text) {
    throwHttp(400, "text is required");
  }

  const reference = estimateReferenceNutrition(text);
  if (reference && reference.calories > 0) {
    return normalizeReferenceDietResult(reference, text, body.mealLabel, body.date);
  }

  const provider = getProvider();
  if (!hasProviderKey(provider)) {
    throwHttp(503, `Missing API key for provider: ${provider}`);
  }

  const prompt = buildDietTextPrompt(body.date, body.mealLabel, text);
  const modelText = await callTextModel(provider, prompt);
  const parsed = parseJsonFromText(modelText);
  return calibrateDietTextResult(normalizeResult(parsed, "diet", body.date, modelText), text, body.mealLabel);
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

  const inlineImage = parseDataUrl(await ensureDataUrl(dataUrl));
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
  const requested = Number.isFinite(parsed) && parsed >= 5000 ? parsed : 12000;
  const maxParsed = Number(process.env.AI_TIMEOUT_MAX_MS);
  const max = Number.isFinite(maxParsed) && maxParsed >= 5000 ? maxParsed : 12000;
  return Math.min(requested, max);
}

function parseSizeToBytes(value, fallback) {
  const text = String(value || "").trim().toLowerCase();
  const match = text.match(/^([0-9]+(?:\.[0-9]+)?)(b|kb|mb|gb)?$/);
  if (!match) return fallback;
  const size = Number(match[1]);
  if (!Number.isFinite(size)) return fallback;
  const unit = match[2] || "b";
  if (unit === "gb") return Math.floor(size * 1024 * 1024 * 1024);
  if (unit === "mb") return Math.floor(size * 1024 * 1024);
  if (unit === "kb") return Math.floor(size * 1024);
  return Math.floor(size);
}

function parseDataUrl(dataUrl) {
  const match = String(dataUrl || "").match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([\s\S]+)$/);
  if (!match) throwHttp(400, "image.dataUrl must be a base64 data URL");
  return {
    mimeType: match[1],
    base64: match[2],
  };
}

function isSupportedImageSource(source) {
  const text = String(source || "");
  return text.startsWith("data:image/") || /^https:\/\/\S+/i.test(text);
}

async function ensureDataUrl(source) {
  const text = String(source || "");
  if (text.startsWith("data:image/")) return text;
  if (!/^https:\/\/\S+/i.test(text)) throwHttp(400, "Unsupported image source");

  const response = await fetchWithHttpError("Image URL", text, {
    method: "GET",
    timeout: getAiTimeoutMs(),
  });
  if (!response.ok) {
    throwHttp(502, `Image URL HTTP ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "image/jpeg";
  const buffer = await response.buffer();
  return `data:${contentType};base64,${buffer.toString("base64")}`;
}

function buildDietTextPrompt(date, mealLabel, text) {
  return [
    "你是个人饮食日志应用的营养估算模块。请根据用户手工描述的食物，估算这一餐的总热量、蛋白、碳水、脂肪，并返回严格 JSON，不要 Markdown。",
    `默认日期: ${date || ""}`,
    `餐别: ${mealLabel || "饮食"}`,
    `用户描述: ${text}`,
    "要求：如果份量不明确，请按常见中国日常份量保守估算；不要编造非常精确的数值；warnings 里说明这是估算。",
    "重要校准：主食不能低估。贝果/面包通常约 250-280 kcal/100g；70g 贝果通常约 180 kcal。熟米饭约 116 kcal/100g，鸡蛋约 70 kcal/个。",
    "meals 字段请写成适合放进饮食日志的一行摘要，包含餐别和主要食物。",
    "JSON schema:",
    JSON.stringify({
      type: "diet",
      date: "YYYY-MM-DD",
      confidence: 0.0,
      summary: "short human readable summary",
      warnings: ["string"],
      rawText: "user food text",
      draft: {
        date: "YYYY-MM-DD",
        diet: {
          calories: null,
          protein: null,
          carbs: null,
          fat: null,
          meals: "",
        },
        notes: "",
      },
    }),
  ].join("\n");
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

const DIET_REFERENCE_FOODS = [
  { aliases: ["贝果", "bagel"], per100g: { calories: 260, protein: 10, carbs: 52, fat: 2 } },
  { aliases: ["吐司", "面包"], per100g: { calories: 265, protein: 9, carbs: 49, fat: 3.5 } },
  { aliases: ["米饭", "白米饭", "熟米饭"], per100g: { calories: 116, protein: 2.6, carbs: 25.6, fat: 0.3 } },
  { aliases: ["鸡胸", "鸡胸肉"], per100g: { calories: 165, protein: 31, carbs: 0, fat: 3.6 } },
  { aliases: ["牛奶"], per100g: { calories: 54, protein: 3.3, carbs: 5, fat: 3.2 } },
  { aliases: ["香蕉"], per100g: { calories: 93, protein: 1.2, carbs: 22, fat: 0.2 } },
  { aliases: ["苹果"], per100g: { calories: 53, protein: 0.3, carbs: 14, fat: 0.2 } }
];

const DIET_REFERENCE_UNITS = [
  { aliases: ["鸡蛋", "蛋"], units: ["个", "颗", "枚"], each: { calories: 70, protein: 6.3, carbs: 0.6, fat: 4.8 } }
];

function calibrateDietTextResult(result, text, mealLabel) {
  const reference = estimateReferenceNutrition(text);
  if (!reference || reference.calories <= 0) return result;

  const draft = result.draft || {};
  const diet = draft.diet || {};
  const modelCalories = Number(diet.calories);
  const shouldUseReference = !Number.isFinite(modelCalories) || modelCalories < reference.calories * 0.65;
  if (!shouldUseReference) return result;

  const next = {
    ...result,
    warnings: [
      ...new Set([...(result.warnings || []), `已按常见食物营养表校准：${reference.evidence.join("、")}`])
    ],
    summary: `${mealLabel || "饮食"}约 ${Math.round(reference.calories)} kcal`,
    draft: {
      ...draft,
      diet: {
        ...diet,
        calories: roundMacro(reference.calories),
        protein: roundMacro(reference.protein),
        carbs: roundMacro(reference.carbs),
        fat: roundMacro(reference.fat),
        meals: diet.meals || `${mealLabel || "饮食"}: ${text}`
      }
    }
  };
  return next;
}

function normalizeReferenceDietResult(reference, text, mealLabel, date) {
  const label = mealLabel || "饮食";
  return normalizeResult({
    type: "diet",
    confidence: 0.92,
    summary: `${label}约 ${Math.round(reference.calories)} kcal`,
    warnings: [`已按常见食物营养表估算：${reference.evidence.join("、")}`],
    rawText: text,
    draft: {
      date,
      diet: {
        calories: roundMacro(reference.calories),
        protein: roundMacro(reference.protein),
        carbs: roundMacro(reference.carbs),
        fat: roundMacro(reference.fat),
        meals: `${label}: ${text}`
      }
    }
  }, "diet", date, text);
}

function estimateReferenceNutrition(text) {
  const value = String(text || "").replace(/\s+/g, "");
  const total = { calories: 0, protein: 0, carbs: 0, fat: 0, evidence: [] };

  for (const food of DIET_REFERENCE_FOODS) {
    const alias = food.aliases.find((item) => value.includes(item.toLowerCase()) || value.includes(item));
    if (!alias) continue;
    const grams = parseFoodGrams(value, food.aliases);
    if (!grams) continue;
    addPer100g(total, food.per100g, grams);
    total.evidence.push(`${grams}g${alias}`);
  }

  for (const food of DIET_REFERENCE_UNITS) {
    const alias = food.aliases.find((item) => value.includes(item));
    if (!alias) continue;
    const count = parseFoodCount(value, food.aliases, food.units);
    if (!count) continue;
    addEach(total, food.each, count);
    total.evidence.push(`${count}${food.units[0]}${alias}`);
  }

  return total.calories > 0 ? total : null;
}

function parseFoodGrams(text, aliases) {
  for (const alias of aliases) {
    const patterns = [
      new RegExp(`([0-9]+(?:\\.[0-9]+)?)\\s*(?:g|克)${escapeRegex(alias)}`, "i"),
      new RegExp(`${escapeRegex(alias)}([0-9]+(?:\\.[0-9]+)?)\\s*(?:g|克)`, "i"),
    ];
    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match) return Number(match[1]);
    }
  }
  return null;
}

function parseFoodCount(text, aliases, units) {
  for (const alias of aliases) {
    for (const unit of units) {
      const patterns = [
        {
          pattern: new RegExp(`([0-9]+(?:\\.[0-9]+)?)${unit}${escapeRegex(alias)}`),
          parse: Number,
        },
        {
          pattern: new RegExp(`${escapeRegex(alias)}([0-9]+(?:\\.[0-9]+)?)${unit}`),
          parse: Number,
        },
        {
          pattern: new RegExp(`([一二两三四五六七八九十半]+)${unit}${escapeRegex(alias)}`),
          parse: parseChineseNumber,
        },
        {
          pattern: new RegExp(`${escapeRegex(alias)}([一二两三四五六七八九十半]+)${unit}`),
          parse: parseChineseNumber,
        },
      ];
      for (const item of patterns) {
        const match = text.match(item.pattern);
        if (match) return item.parse(match[1]);
      }
    }
  }
  return null;
}

function parseChineseNumber(value) {
  const text = String(value || "");
  if (text === "半") return 0.5;
  const digitMap = {
    一: 1,
    二: 2,
    两: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
  };
  if (!text.includes("十")) return digitMap[text] || null;

  const [tensText, onesText] = text.split("十");
  const tens = tensText ? digitMap[tensText] : 1;
  const ones = onesText ? digitMap[onesText] : 0;
  if (!tens || ones === undefined) return null;
  return tens * 10 + ones;
}

function addPer100g(total, per100g, grams) {
  const factor = grams / 100;
  total.calories += per100g.calories * factor;
  total.protein += per100g.protein * factor;
  total.carbs += per100g.carbs * factor;
  total.fat += per100g.fat * factor;
}

function addEach(total, each, count) {
  total.calories += each.calories * count;
  total.protein += each.protein * count;
  total.carbs += each.carbs * count;
  total.fat += each.fat * count;
}

function roundMacro(value) {
  return Math.round(Number(value) * 10) / 10;
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
