/** @format */

import {
  JsonOutputParser,
  StructuredOutputParser,
} from "@langchain/core/output_parsers";
import { model } from "../../configs/model.config.js";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda } from "@langchain/core/runnables";
import { z } from "zod";
import { OutputFixingParser } from "@langchain/classic/output_parsers";

// create a RunnableLamda function to validate LLM response
function validateLLMResponse(schema) {
  console.log("⏳ Schema validation started...");
  return RunnableLambda.from((input) => {
    const validate = schema.safeParse(input);
    const { success: isSuccess, error } = validate;

    if (!isSuccess) {
      console.log("❌ Schema validation failed");
      const issues = error?.issues ?? [];
      const errorMsg = issues.map(
        (issue, idx) => `${idx + 1}. ${issue.message}`
      );
      throw new Error(`${errorMsg}`);
    }
    console.log("✅ Schema validation successfull...");
    return validate?.data;
  });
}

// create a helper function to invoke LLM with retries mechanisum
async function callLLMWithRetries(chain, input, retries = 3, data_key) {
  if (!chain) throw new Error("Provided chain is invalid...");
  else if (!input) throw new Error("Provided inputs is invalid...");
  else if (!data_key) throw new Error("Provided data_key is invalid...");
  let lastError = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      let enrichInput = lastError
        ? {
            ...input,
            [data_key]:
              input[data_key] +
              `PREVIOUS ATTEMPT ERROR: ${lastError}. Please fix and try again`,
          }
        : input;
      // console.log(enrichInput);
      const result = await chain(enrichInput);
      if (attempt > 1) console.log(`✅ Succeeded on attempt ${attempt}`);
      return result;
    } catch (error) {
      lastError = error.message;
      const delay = Math.pow(2, attempt) * 100;
      console.log(`⏳ Retrying in ${delay}ms...`);
      await new Promise((res) => setTimeout(res, delay));
    }
  }

  throw new Error(
    `Failed for all ${retries} attempts, with last error ${lastError}`
  );
}

/**
 * ✅ Question 1 — User Profile Extraction
 * Problem Description: A user onboarding service receives free-text "about me" blurbs submitted during sign-up. The backend must extract a structured user profile (name, email, age) before writing to the database. Raw LLM output may omit fields or use wrong types, so Zod validation is required to catch bad data before it reaches the DB. Email format and age range are enforced at the schema level so invalid records are rejected immediately.
 */
async function pattern_01() {
  console.time("⏳ Execution Time");
  console.log("=== Question-1: User Profile Extraction ===");
  const rawData = `
      Hey there! I'm Alex Johnson, you can reach me at alex.johnson@company.com.
      I'm 34 years old and just joined your platform to track my fitness goals.
    `;
  try {
    // define prompt for LLM
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a user profile extractor. Extract user profile detail information and return valid JSON object no markdown, no code fences, no explaination.
        RULES:
          - Do not hallucinate values
          - If any value is unclear return "", do not return values like "Unknown", "Invalid" or null
        Return JSON object should contain following fields:
         - name : {{ Name should be string Ex. Jane Dow }}
         - email: {{ Email should be string Ex. jane@example.com }}
         - age: {{ Age should be number Ex. 18 }}
         - summary: {{ Summarize user message within maximum 15 characters }}
         {format_instructions}`,
      ],
      ["human", "Extract user information from\n\n{rawData}"],
    ]);
    // define schema to validate response
    const schema = z.object({
      name: z
        .string()
        .trim()
        .min(1, "User name cannot be empty")
        .describe("Name of the user"),
      email: z
        .string()
        .trim()
        .email("Invalid email format")
        .min(1, "User email cannot be empty")
        .describe("Name of the user"),
      age: z
        .number()
        .int("User age should be integer")
        .positive("Age cannot be negative")
        .min(18, "User must be at least 18")
        .max(120, "Age seems unrealistic"),
      summary: z.string().trim().min(1, "User message summary cannot be empty"),
    });
    // create a helper RunnableLamda function to validate LLM response with schema
    const validateResponse = validateLLMResponse(schema);
    // declear parser
    const parser = StructuredOutputParser.fromZodSchema(schema);
    // create a chain prompt → LLM → parser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateResponse);
    // invoke chain with retries mechanisum
    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        rawData,
      },
      3,
      "rawData"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_01();

/**
 * ✅ Question 2 — Product Specification Parser
 * Problem Description: An e-commerce platform allows vendors to paste unstructured product descriptions. The system must extract a clean product specification (title, price, currency, sku, category, summary, features, in_stock) to populate the catalog. Zod ensures price is a positive number, currency is a valid 3-letter ISO code, and features contains at least one entry before the record is stored. The in_stock field must be a proper boolean — not a string like "yes" or "true".
 */
async function pattern_02() {
  console.time("⏳ Execution Time");
  console.log("\n=== Question-2: Product Specification Parser ===\n");

  const rawDescription = `
      Introducing the UltraSound Pro X9 wireless speaker!
      Price: $249.99 USD. Currently in stock.
      - 360° surround sound with 40W output
      - 24-hour battery life
      - IPX7 waterproof rating
      - Bluetooth 5.3 & AUX input
      - Available in Midnight Black and Arctic White
      SKU: USP-X9-BLK
      Category: Consumer Electronics > Audio
    `;
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a Product Specification Parser. Extract product information from given raw test and return valid JSON object with no code fences, no markdown, no explanation
        RULES:
          - Never hallucinate values
          - if any value is unclear return "" never return "Unknown", "Invalid"
        Return JSON object should contain following fields:
          - product_name: {{ Product name should be string, Ex. Iphone-16 }}
          - product_price: {{ Product price in USD should be string Ex. $249.99 USD  }}
          - in_stock: {{ In stock should be boolean value, if stock is available then true otherwise false}}
          - category: {{ Product category shuld be string Ex. Mobile }}
          - features: {{ Array of string - minimum 1 feature is required }}
          {format_instructions}`,
      ],
      ["human", "Extract product information from \n\n{product_details}"],
    ]);
    // create a zod schema
    const schema = z.object({
      product_name: z.string().trim().min(1, "Product name cannot be empty"),
      product_price: z.string().trim().min(1, "Product price cannot be empty"),
      in_stock: z.boolean(),
      category: z.string().trim().min(1, "Product category cannot be empty"),
      features: z
        .array(z.string().trim().min(1))
        .min(1, "Atleast 1 feature is needed"),
    });
    // initiate parser
    const parser = StructuredOutputParser.fromZodSchema(schema);
    // validate LLM respose using custom RunnableLamda fn
    const validateSchema = validateLLMResponse(schema);
    // create a chain using prompt, model & parser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateSchema);
    // call LLM with retry mechanisum
    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        product_details: rawDescription,
      },
      3,
      "product_details"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_02();

/**
 * ✅ Question 3 — Support Ticket Sentiment Classifier
 * Problem Description: A customer support analytics platform needs to classify incoming support tickets by sentiment (positive / neutral / negative), extract a short summary (max 200 characters), assign a priority score (1–5), and flag whether escalation is needed. Zod enforces the enum constraint on sentiment and the integer range on priority so downstream ticket-routing rules never receive unexpected values. The chain must reject any output where sentiment falls outside the three allowed values or priority is not a whole number in the valid range.
 */
async function pattern_03() {
  console.time("⏳ Execution Time");
  console.log("=== Question-3: Support Ticket Sentiment Classifier ===");
  const ticketText = `
      I placed order #98123 over three weeks ago and it still hasn't arrived.
      I've emailed support twice with zero response. This is completely unacceptable
      and I want a full refund immediately!
    `;
  try {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a Support Ticket Sentiment Classifier. Extract ticket information and return valid JSON object no markdown, no code fences, no explanation.
        RULES:
          - Never hallucinate values
          - If any value is unclear then return "", never return "Invalid", "Unknown".
        Return JSON object should have these following fields:
          - sentiment: {{ sentiment should be string and value of the field should be any one of the positive / neutral / negative }}
          - summary: {{ summarize customer message maximum 50 character }}
          - priority_score: {{ It should be a number from 1-5, 1=Lowest priority & 5=Highest priority }}
          - escalation: {{ It should be a boolean value, if priority_score is high then it should be true }}
        {format_instructions}`,
      ],
      ["human", "Extract information from\n\n{ticket_details}"],
    ]);
    const schema = z.object({
      sentiment: z.enum(["positive", "neutral", "negative"], {
        errorMap: () => ({
          message: "Sentiment value should be positive / neutral / negative ",
        }),
      }),
      summary: z
        .string()
        .trim()
        .min(1, "Summary of customer message cannot be empty"),
      priority_score: z
        .number()
        .int("priority_score should be integer")
        .min(1, "priority_score cannot be negative or 0")
        .max(5, "priority_score should be within 5"),
      escalation: z.boolean(),
    });
    const parser = StructuredOutputParser.fromZodSchema(schema);
    const validateSchema = validateLLMResponse(schema);

    const chain = prompt.pipe(model).pipe(parser).pipe(validateSchema);
    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        ticket_details: ticketText,
      },
      3,
      "ticket_details"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_03();

/**
 * ✅ Question 4 — Job Posting Nested Object Extractor
 * Problem Description: A job board scraper processes raw job-posting text scraped from the web. Each posting must be parsed into a nested structure: a company object (name, location, remote flag) and a job object (title, salary_min, salary_max, skills array). Nested Zod objects validate the shape recursively. A .refine() cross-field check enforces that salary_max >= salary_min so logically invalid ranges are caught before insertion into the search index. The skills array must contain at least one entry.
 */
async function pattern_04() {
  console.time("⏳ Execution Time");
  console.log("=== Question-4: Job Posting Nested Object Extractor ===");
  const rawPosting = `
      🚀 Acme Corp is hiring! (San Francisco, USA — Remote OK)
 
      Role: Senior TypeScript Engineer
      Compensation: $160,000 – $200,000 per year
 
      What you'll need:
      • TypeScript (5+ years)
      • Node.js & Express
      • PostgreSQL and Redis
      • Docker & Kubernetes
 
      Apply by sending your CV to careers@acme.com
    `;
  try {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a job board scraper processes. Extract job details from given data and return valid JSON object no code fences, no explanation and no markdown.
        RULES:
          - Never hallicinate vallues
          - If any one of the value is unclear then return "", never return "Unknown" or "Invalid"
        Return valid JSON object should have following fields:
          - company_details: {{ name should be string, location should be string and remote_flag should be boolean }}
          - job_details: {{ title should be string, salary_min should be number, salary_max should be number, skills array of strings }}
          -
          {format_instructions}`,
      ],
      ["human", "Extract information from\n\n{job_details}"],
    ]);
    const schema = z.object({
      company_details: z.object({
        name: z.string().trim().min(1, "Company name cannot be empty"),
        location: z.string().trim().min(1, "Company location cannot be empty"),
        remote_flag: z.boolean(),
      }),
      job_details: z.object({
        title: z.string().trim().min(1, "Job title cannot be empty"),
        salary_min: z
          .number()
          .int("Minimum salary should be integer")
          .positive("Minimum salary cannot be negative"),
        salary_max: z
          .number()
          .int("Maximum salary should be integer")
          .positive("Maximum salary cannot be negative"),
        skills: z
          .array(z.string().trim().min(1))
          .min(1, "Atleast 1 skill is required"),
      }),
    });
    const parser = StructuredOutputParser.fromZodSchema(schema);
    const validateResponse = validateLLMResponse(schema);

    const chain = prompt.pipe(model).pipe(parser).pipe(validateResponse);
    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        job_details: rawPosting,
      },
      3,
      "job_details"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_04();

/**
 * ❌ Question 5 — Recipe Normaliser with Optional Fields
 * Problem Description: A recipe management app lets users paste cooking instructions in any free-text format. The system must normalise them into a structured object: name, servings, an ingredients array (each with name, quantity, unit), and ordered steps. prep_time_minutes and cook_time_minutes are optional — captured when present but the schema must not fail when absent. Zod's .optional().nullable() handles these gracefully. quantity must always be a positive number and steps must have at least one entry.
 */
async function pattern_05() {
  console.time("⏳ Execution Time");
  console.log("=== Question-5: Recipe Normaliser with Optional Fields ===");
  const rawRecipeText = `
      Classic Banana Pancakes — serves 2
 
      You'll need: 1 ripe banana, 2 large eggs, half a cup of rolled oats,
      a pinch of cinnamon, and a teaspoon of vanilla extract.
 
      Heat a non-stick pan over medium heat and lightly grease it.
      Mash the banana in a bowl until smooth, then whisk in the eggs.
      Stir in the oats, cinnamon, and vanilla.
      Pour small rounds of batter into the pan and cook about 2 minutes per side
      until golden. Serve warm with honey or fruit.
 
      Total time roughly 15 minutes.
    `;
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_05();

/**
 *  Question 6 — Financial News Classifier with Regex Constraints
 * Problem Description: A financial news aggregator must classify each article with: a list of stock ticker symbols (uppercase, 1–5 chars, enforced by .regex()), a market impact enum (bullish / bearish / neutral), a confidence score clamped to [0, 1], and an optional array of mentioned executives with their roles. Zod's.regex() on each ticker and .min()/.max() on the confidence float prevent invalid data from reaching trading algorithms downstream. The executives field is optional and should only appear when executives are explicitly named in the article.
 */
async function pattern_06() {
  console.time("⏳ Execution Time");
  console.log(
    "=== Question-6: Financial News Classifier with Regex Constraints ==="
  );
  const newsArticle = `
      Apple (AAPL) and Microsoft (MSFT) both surged in after-hours trading on Tuesday
      after Apple CEO Tim Cook announced record-breaking Q2 revenue of $94.8 billion,
      beating Wall Street estimates by a wide margin. Microsoft CFO Amy Hood also
      commented that Azure cloud growth remains robust heading into Q3.
      Analysts widely expect the rally to continue through the week.
    `;
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a financial news analyser. Return a JSON object only — no markdown, no code fences, no explanation.
        RULES:
          - Do not hallucinate values
          - tickers must be 1–5 uppercase letters only (e.g. AAPL, MSFT)
          - impact must be exactly one of: "bullish", "bearish", "neutral"
          - confidence must be a decimal between 0 and 1 (e.g. 0.85)
          - only include executives if they are explicitly named in the article
        Return values should contain:
          - tickers: {{Array of stock ticker strings}}
          - impact: {{"bullish", "bearish", or "neutral"}}
          - confidence: {{Confidence score 0.0–1.0}}
          - executives: {{Optional array of {{ name (string), role (string) }} — omit if none mentioned}}
        {format_instructions}`,
      ],
      ["human", "Analyse this financial news article:\n\n{article}"],
    ]);
    const schema = z.object({
      tickers: z
        .array(
          z
            .string()
            .regex(
              /^[A-Z]{1,5}$/,
              "Each ticker must be 1–5 uppercase letters only"
            )
        )
        .min(1, "At least one ticker must be identified"),
      impact: z.enum(["bullish", "bearish", "neutral"], {
        errorMap: () => ({
          message: "Impact must be bullish, bearish, or neutral",
        }),
      }),
      confidence: z
        .number()
        .max(1, "Maximum value of confidence is 1")
        .min(0, "Minimum value of confidence is 0"),
      executives: z
        .array(
          z.object({
            name: z.string().trim().min(1, "Executive name cannot be empty"),
            role: z.string().trim().min(1, "Executive role cannot be empty"),
          })
        )
        .optional(),
    });
    const parser = StructuredOutputParser.fromZodSchema(schema);
    const validateResponse = validateLLMResponse(schema);
    const chain = prompt.pipe(model).pipe(parser).pipe(validateResponse);
    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        article: newsArticle,
      },
      3,
      "article"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_06();

/**
 * ✅ Question 7 — Legal Contract Key-Term Extractor
 * Problem Description: A legal-tech platform processes contract excerpts and must extract: contracting parties (array of {name, role enum}), effective date (validated as YYYY-MM-DD by regex), governing law jurisdiction, termination conditions (string array, min 1), and optional penalty clauses ({trigger,amount_usd}). Contracts vary wildly in language; Zod's regex on dates and cross-validated role enums ensure all extracted data is machine-readable before it enters the contract database. Penalty clauses are optional and omitted when none are present in the source text.
 */
async function pattern_07() {
  console.time("⏳ Execution Time");
  console.log("=== Question-7: Legal Contract Key-Term Extractor ===");
  const contractExcerpt = `
      SERVICE AGREEMENT
 
      This Agreement is entered into as of January 15, 2025, by and between
      Acme Corporation ("Vendor") and Beta Solutions LLC ("Client").
 
      Governing Law: This Agreement shall be governed by the laws of New York, USA.
 
      Term & Termination: This Agreement may be terminated by either party upon
      90 days' written notice, or immediately upon material breach by either party,
      or by mutual written consent of both parties.
 
      Penalties: In the event of a confidentiality breach, the breaching party shall
      pay liquidated damages of $50,000 USD. Late payment beyond 30 days incurs a
      penalty of $5,000 USD per month.
    `;
  try {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a Legal Contract Key-Term Extractor. Extract information and return valid JSON format no code fences, no explanations and no markdown.
        RULES:
          - Never hallucinate values
          - If any one of the value is unclear then return "", never return values like "Unknown", "Invalid"
        Return JSON object must contain following fields:
          - parties: {{ Array of name(must be string) and role(enum) -- must have atleast 2 }}
          - effective_date: {{ String of data format should be YYYY-MM-DD }}
          - governing_law: {{ Jurisdiction string e.g. "New York, USA" }}
          - termination_conditions: {{ Array of string -- must have atleast 1 }}
          - penalty_clauses: {{ Array of trigger(string), amount_usd(number) — omit if none }}
        {format_instructions}`,
      ],
      ["human", "Extract information from\n\n{contract_details}"],
    ]);
    const schema = z.object({
      parties: z
        .array(
          z.object({
            name: z.string().trim().min(1, "Party name is required"),
            role: z.enum(
              [
                "client",
                "vendor",
                "contractor",
                "employer",
                "employee",
                "other",
              ],
              {
                errorMap: () => ({
                  message:
                    "role must be client/vendor/contractor/employer/employee/other",
                }),
              }
            ),
          })
        )
        .min(2, "Atleast 2 parties is required"),
      effective_date: z
        .string()
        .regex(/^\d{4}-\d{2}-\d{2}$/, "Date format is invalid"),
      governing_law: z.string().trim().min(1, "Governing law cannot be empty"),
      termination_conditions: z
        .array(z.string().trim().min(1))
        .min(1, "Termination conditions cannot be empty"),
      penalty_clauses: z
        .array(
          z.object({
            trigger: z
              .string()
              .trim()
              .min(1, "Penalty trigger cannot be empty"),
            amount_usd: z.number().positive("Amount value should be positive"),
          })
        )
        .optional(),
    });
    const parser = StructuredOutputParser.fromZodSchema(schema);
    const validateResponse = validateLLMResponse(schema);
    const chain = prompt.pipe(model).pipe(parser).pipe(validateResponse);

    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        contract_details: contractExcerpt,
      },
      3,
      "contract_details"
    );
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_07();

/**
 * Question 8 — Medical Transcription with Self-Healing Parser (OutputFixingParser)
 * Problem Description: A medical transcription service converts doctor–patient dialogue into structured clinical notes. The deeply nested schema includes patient vitals (blood pressure, heart rate, temperature, SpO2), a diagnoses array (each with a validated ICD-10 code matching /^[A-Z]\d{2}(\.\d+)?$/ and severity enum), medications (name, dose_mg, frequency, optional duration), and follow-up instructions. LLMs occasionally return malformed JSON for complex nested schemas. The chain uses OutputFixingParser wrapping StructuredOutputParser to automatically repair bad output via a second LLM call (the self-healing pattern), followed by a Zod safeParse validation layer.
 */
async function pattern_08() {
  console.time("⏳ Execution Time");
  console.log(
    "=== Question-8: Medical Transcription with Self-Healing Parser ==="
  );
  const doctorPatientTranscript = `
      Doctor: Good morning. Let me check your vitals first.
              Blood pressure 138/86 mmHg, heart rate 82 bpm,
              temperature 37.4°C, oxygen saturation 96%.
 
      Patient: I've had a bad sore throat and fever for three days.
 
      Doctor: I can see the throat is quite inflamed. Based on the presentation I'm
              diagnosing acute tonsillitis — that's J03.90 — moderate severity,
              and also mild dehydration, E86.0.
 
              I'm prescribing Amoxicillin 500mg three times daily for 10 days,
              and Ibuprofen 400mg every 8 hours as needed for pain and fever —
              take that for 5 days maximum.
 
              Drink plenty of fluids. Come back in one week if you're not improving,
              or sooner if you develop difficulty breathing or swallowing.
    `;
  try {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a medical transcription AI. Extract a structured clinical note from the
        conversation. Return valid JSON only — no markdown, no code fences, no explanation.
        RULES:
          - Do not hallucinate clinical values
          - ICD-10 codes must be one uppercase letter followed by two digits, optionally
            a dot and more digits (e.g. J03.9, E86.0)
          - severity must be exactly: "mild", "moderate", or "severe"
        `,
      ],
      [
        "human",
        "Transcribe this conversation into a clinical note:\n\n{transcript}",
      ],
    ]);
    const schema = z.object({
      patient: z.object({
        vitals: z.object({
          blood_pressure: z
            .string()
            .trim()
            .min(1, "Blood pressure cannot be empty"),
          heart_rate_bpm: z
            .number()
            .int()
            .positive("Heart rate must be a positive integer"),
          temperature_c: z.number().describe("Body temperature in Celsius"),
          spo2_percent: z
            .number()
            .min(0, "SpO2 cannot be below 0")
            .max(100, "SpO2 cannot exceed 100"),
        }),
      }),
      diagnoses: z
        .array(
          z.object({
            description: z
              .string()
              .trim()
              .min(1, "Diagnosis description cannot be empty"),
            icd10_code: z
              .string()
              .regex(
                /^[A-Z]\d{2}(\.\d{1,2})?$/,
                "Must be a valid ICD-10 code e.g. J06.9"
              ),
            severity: z.enum(["mild", "moderate", "severe"], {
              errorMap: () => ({
                message: "Severity must be mild, moderate, or severe",
              }),
            }),
          })
        )
        .min(1, "At least one diagnosis must be recorded"),
      medications: z
        .array(
          z.object({
            name: z.string().trim().min(1, "Medication name cannot be empty"),
            dose_mg: z.number().positive("Dose must be a positive number"),
            frequency: z.string().trim().min(1, "Frequency cannot be empty"),
            duration_days: z.number().int().positive().optional(),
          })
        )
        .min(1, "At least one medication must be prescribed"),
      follow_up: z
        .string()
        .trim()
        .min(1, "Follow-up instructions cannot be empty"),
    });
    const baseParser = StructuredOutputParser.fromZodSchema(schema);
    const fixingParser = OutputFixingParser.fromLLM(model, baseParser);

    const validateResponse = validateLLMResponse(schema);
    const chain = prompt.pipe(model).pipe(fixingParser).pipe(validateResponse);

    // const response = await callLLMWithRetries(
    //   (input) => chain.invoke(input),
    //   {
    //     format_instructions: baseParser.getFormatInstructions(),
    //     transcript: doctorPatientTranscript,
    //   },
    //   3,
    //   "transcript"
    // );
    const response = await chain.invoke({
      format_instructions: baseParser.getFormatInstructions(),
      transcript: doctorPatientTranscript,
    });
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_08();

/**
 * Question 9 — Batch Review Analytics (Single Prompt, Array Output)
 * Problem Description: A SaaS analytics pipeline must process an array of raw customer reviews and return structured sentiment data for every review. Rather than calling the LLM N times (expensive and slow), a single prompt batches all eviews and the LLM returns an array of structured results wrapped in a results key. Zod validates the outer wrapper and each element — enforcing sentiment enum, score range [0–10], non-empty topics array, and a boolean recommended flag. A graceful degradation fallback ensures one bad parse does not crash the entire batch; failed batches are replaced with safe neutral defaults per review.
 * review_id, sentiment, score, topics, recommended
 */
async function pattern_09() {
  console.time("⏳ Execution Time");
  const customerReviews = [
    {
      id: "r1",
      text: "Absolutely love this product! Best purchase I've made this year. Fast delivery too.",
    },
    {
      id: "r2",
      text: "Arrived completely broken. Support ignored my emails for two weeks. Never buying again.",
    },
    {
      id: "r3",
      text: "Decent quality for the price. Nothing revolutionary but gets the job done.",
    },
    {
      id: "r4",
      text: "Exceeded every expectation. The build quality is outstanding and battery life is incredible.",
    },
  ];
  try {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a Review analizer. Analize customer reviews and return valid JSON object no code fences, no explanations and no markdown
        RULES:
          - Never hallucinate values
          - If any value is unclear return "", never return "Invalid", "Unknown"
        Return object must have following fields:
          - review_id: {{ String - must have review ID }}
          - sentiment: {{ sentiment(enum) (positive/negative/neutral) }}
          - score: {{ number(range of the score is 0-10s) }}
          - topics: {{ Array(string) - must have atleast 1 }}
          - recommended: {{ boolean }}
          {format_instructions}`,
      ],
      ["human", "Extract information\n\n{customer_review}"],
    ]);
    const singleReviewSchema = z.object({
      review_id: z.string().trim().min(1, "Review ID cannot be empty"),
      sentiment: z.enum(["positive", "negative", "neutral"], {
        errorMap: () => ({
          message: "Sentiment must be positive, negative, neutral",
        }),
      }),
      score: z
        .number()
        .positive("score cannot be negative")
        .max(10, "Maximum value of score is 10"),
      topics: z.array(z.string().trim().min(1)).min(1, "Atleast 1 is required"),
      recommended: z.boolean(),
    });
    const allReviewsSchema = z.object({
      results: z
        .array(singleReviewSchema)
        .min(1, "atleast one review is required"),
    });
    const parser = StructuredOutputParser.fromZodSchema(allReviewsSchema);
    const validateResponse = validateLLMResponse(allReviewsSchema);
    const chain = prompt.pipe(model).pipe(parser).pipe(validateResponse);

    const response = await callLLMWithRetries(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        customer_review: customerReviews,
      },
      3,
      "customer_review"
    );
    console.log("✅ RESPONSE:", JSON.stringify(response, 2, null));
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
// pattern_09();

/**
 * Question 10 — Multi-Step Competitive Intelligence Pipeline
 * Problem Description: A competitive intelligence platform runs a three-stage
 * research pipeline on a raw company description:
 *   Step 1 — Extract structured company facts (name, founded_year, headquarters,
 *             products array, optional employee count). Year range and products
 *             array are validated by Zod.
 *   Step 2 — Generate a SWOT analysis from Step 1 output. Each quadrant is a
 *             non-empty string array validated individually by Zod.
 *   Step 3 — Score investment attractiveness (integer 0–100), produce a rationale
 *             string (min 50 chars), and emit a recommendation enum.
 * Each step has its own prompt, JsonOutputParser, Zod schema, and independent
 * try/catch block so a failure in Step 2 does not discard the already-validated
 * Step 1 output. The function returns a typed composite intelligence report.
 */
async function pattern_10() {
  console.time("⏳ Execution Time");
  const companyBlurb = `
      Stripe, Inc. was founded in 2010 and is headquartered in San Francisco, CA.
      The company offers a suite of financial infrastructure products including payment
      processing APIs, Stripe Atlas (business incorporation), Radar (ML fraud detection),
      Stripe Connect (marketplace payments), and Stripe Treasury (banking-as-a-service).
      The company employs approximately 8,000 people globally and processes hundreds of
      billions of dollars in payments annually across more than 40 countries.
    `;
  try {
    // ── STEP 1: Extract Company Facts ─────────────────────────────
    console.log("=== STEP 1: Extract Company Facts ===");
    const prompt_1 = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a extractor. Extract company information from given raw text and return valid JSON object with no code fences, no explanation and no markdown
        RULES:
          - Never hallucinate values
          - If any value is unclear then just return "", never return values like 'Unknown" or "Invalid"
        Return JSON object must contain following fields:
          - name: {{ Company name should be string }},
          - founded_year: {{ Company founded year - Company found date cannot be future date }}
          - headquarters: {{ headquarters should be string }},
 *        - products: {{ Array of string }},
          - employee: {{ Number of employee - omit 0 if not found }}
          {format_instructions}`,
      ],
      ["human", "Extract company details from \n\n{company_details}"],
    ]);
    const schema_1 = z.object({
      name: z.string().trim().min(1, "Company name cannot be empty"),
      founded_year: z
        .number()
        .int("Founded year must be a whole number")
        .min(1800, "Founded year seems too early")
        .max(new Date().getFullYear(), "Cannot set future date"),
      products: z
        .array(z.string().trim().min(1))
        .min(1, "Atleast 1 product is required"),
      employee: z.number().optional(),
    });
    const validateResponse = validateLLMResponse(schema_1);
    const parser_1 = StructuredOutputParser.fromZodSchema(schema_1);
    const chain_1 = prompt_1.pipe(model).pipe(parser_1).pipe(validateResponse);

    const response_1 = await callLLMWithRetries(
      (input) => chain_1.invoke(input),
      {
        format_instructions: parser_1.getFormatInstructions(),
        company_details: companyBlurb,
      },
      3,
      "company_details"
    );
    console.log("\n✅ Response from pipeline-1:");
    console.log(response_1);
    console.log("=".repeat(80) + "\n");

    // ── STEP 2: SWOT Analysis ──────────────────────────────────────
    console.log("=== STEP 2: SWOT Analysis ===");
    const prompt_2 = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a SWOT Analyzer. Analyze company details and return valid JSON object, no markdown, no code fences, no explanation
        RULES:
          - Never hallucinate values
          - If any value is unclear then return "", never return values like "Invalid", "Unknown"
        Return JSON object must have following fields:
          - trengths (string array)
          - weaknesses (string array),
          - opportunities (string array)
          - threats (string array)
        {format_instructions}`,
      ],
      ["human", "Extract information from\n\n{company_json}"],
    ]);
    const schema_2 = z.object({
      trengths: z
        .array(z.string().trim().min(1))
        .min(1, "Trengths cannot be empty"),
      weaknesses: z
        .array(z.string().trim().min(1))
        .min(1, "Weaknesses cannot be empty"),
      opportunities: z
        .array(z.string().trim().min(1))
        .min(1, "Opportunities cannot be empty"),
      threats: z
        .array(z.string().trim().min(1))
        .min(1, "Threats cannot be empty"),
    });
    const parser_2 = StructuredOutputParser.fromZodSchema(schema_2);
    const validateResponse_2 = validateLLMResponse(schema_2);
    const chain_2 = prompt_2
      .pipe(model)
      .pipe(parser_2)
      .pipe(validateResponse_2);
    const response_2 = await callLLMWithRetries(
      (input) => chain_2.invoke(input),
      {
        format_instructions: parser_2.getFormatInstructions(),
        company_json: response_1,
      },
      3,
      "company_json"
    );
    console.log("\n✅ Response from pipeline-1:");
    console.log(response_2);
    console.log("=".repeat(80) + "\n");

    // ── STEP 3: Investment Scoring ─────────────────────────────────
    // ["strong buy", "buy", "hold", "avoid"]
    console.log("=== STEP 3: Investment Scoring ===");
    const prompt_3 = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a senior investment analyst. Return a JSON object only — no markdown, no code fences, no explanation.
        RULES:
          - score must be an integer from 0 (worst) to 100 (best investment)
          - rationale must be 2–3 complete, informative sentences explaining the score
          - recommendation must be exactly one of: "strong buy", "buy", "hold", "avoid"
        Return values should contain:
          - score (integer 0–100), rationale (string), recommendation (enum)
        {format_instructions}`,
      ],
      [
        "human",
        "Score this company's investment attractiveness.\n\nCompany Facts:\n{company_json}\n\nSWOT:\n{swot_json}",
      ],
    ]);
    const schema_3 = z.object({
      score: z
        .number()
        .int("Score must be a whole number")
        .min(0, "Score cannot be below 0")
        .max(100, "Score cannot exceed 100"),
      rationale: z
        .string()
        .trim()
        .min(
          50,
          "Rationale must be at least 50 characters — write 2–3 full sentences"
        ),
      recommendation: z.enum(["strong buy", "buy", "hold", "avoid"], {
        errorMap: () => ({
          message:
            "Recommendation must be exactly: strong buy, buy, hold, or avoid",
        }),
      }),
    });
    const parser_3 = StructuredOutputParser.fromZodSchema(schema_3);
    const validateResponse_3 = validateLLMResponse(schema_3);
    const chain_3 = prompt_3
      .pipe(model)
      .pipe(parser_3)
      .pipe(validateResponse_3);
    const response_3 = await callLLMWithRetries(
      (input) => chain_3.invoke(input),
      {
        format_instructions: parser_2.getFormatInstructions(),
        company_json: response_1,
        swot_json: response_2,
      },
      3,
      "company_json"
    );
    console.log("\n✅ Response from pipeline-1:");
    console.log(response_3);
    console.log("=".repeat(80) + "\n");
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time");
    process.exit(1);
  }
}
pattern_10();
