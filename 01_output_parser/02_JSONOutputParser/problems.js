/** @format */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { model } from "../../configs/model.config.js";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { email, z } from "zod";
import { RunnableLambda, RunnableRetry } from "@langchain/core/runnables";
import pLimit from "p-limit";

// Call LLM With retry mechnisum
async function callLLMWithRetresMechanisum(
  chain,
  input,
  retries = 3,
  label = "",
  data_key
) {
  let lastErr = null;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      let enrichInput = lastErr
        ? {
            ...input,
            data_key:
              input[data_key] +
              `\n\nPREVIOUS ATTEMP ERROR: ${lastErr}. Please fix and retry`,
          }
        : input;
      const result = await chain(enrichInput);
      if (attempt > 1)
        console.log(`✅ Succeeded on attempt ${attempt} for ${label}`);
      return result;
    } catch (error) {
      lastErr = error.message;
      if (attempt < retries) {
        const delay = Math.pow(2, attempt) * 100;
        console.log(`⏳ Retrying in ${delay}ms...`);
        await new Promise((res) => setTimeout(res, delay));
      }
    }
  }
  throw new Error(
    `❌ All ${retries} attempts failed for [${label}]. Last error: ${lastErr}`
  );
}

/**
 * Question 1 — Basic JSON Output Parsing
 * Problem Description: You are building a simple product information extractor. Given a plain text description of a product, you need the LLM to return structured data in JSON format.
 */
async function pattern_01() {
  console.time("# Execution Time");
  try {
    console.log("\n=== Question-1: Basic JSON Output Parsing ===\n");
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

    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a product details extractor. Return JSON object as response no markdown, no code fences and no explanation.
        RULES:
          - Do not hallucinate value
        Return values should contains:
          - product_name: {{Name of the product}}
          - product_price: {{Price of the product}}
          - currency: {{Product price currency}}
          - product_category: {{Category of product}}
          - product_summary: {{Small product summary within 20-30 words}}
          - product_sku: {{SKU product}}
          - product_features: {{list of all features}}
          - product_stock: {{is product available}}
        {format_instructions}`,
      ],
      ["human", "Extract product information from \n\n{product_details}"],
    ]);

    // declear Json output parser
    const parser = new JsonOutputParser();

    // declear zod schema
    const schema = z.object({
      product_name: z.string().trim().min(1, "Product name cannot be empty"),
      product_price: z
        .number()
        .min(1, "Product price cannot be 0 or negative value"),
      currency: z
        .string()
        .trim()
        .min(1, "Product price currency cannot be empty"),
      product_category: z
        .string()
        .trim()
        .min(1, "Product category cannot be empty"),
      product_summary: z
        .string()
        .trim()
        .min(1, "Product summary cannot be empty"),
      product_sku: z.string().trim().min(1, "Product SKU cannot be empty"),
      product_features: z
        .array(z.string())
        .min(1, "Minimum 1 feature is required for product"),
      product_stock: z.boolean(),
    });

    // create a custom function to validate LLM response
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;

      if (!isSuccess) {
        console.log("\n❌ Schema validation error...\n");
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}\n`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log("✅ Schema validation successfull");
      return validate?.data;
    });

    // create chain prompt → LLM → JsonOutputParser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);

    // invoke LLM & get response
    // const response = await chain.invoke({
    //   format_instructions: parser.getFormatInstructions(),
    //   product_details: rawDescription,
    // });
    const response = await callLLMWithRetresMechanisum(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        product_details: rawDescription,
      },
      3,
      "pattern_1",
      "product_details"
    );
    console.log("✅ RESPONSE:", response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_01();

/**
 * Question 2 — Format Instructions Injection
 * Problem Description: You want the LLM to reliably return JSON without hallucinating extra fields or wrapping the output in markdown code fences.
 */
async function pattern_02() {
  console.time("# Execution Time");
  console.log("\n=== Question-2: Format Instructions Injection ===\n");
  const customerMessage =
    "Hi, my name is Avijit Goswami, I was charged twice for my subscription for avijit@capitalnumbers.com this month and I need an immediate refund. This is unacceptable!";
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a customer message extractor. Extract customer message and return valid JSON object no explanation, no code fences, no markdown.
        RULES:
          - Nevel hallucinate values
          - If value not found never return "Unknown", "Null" or "Invalid". Just return "".
        Return JSON object should have
          - name: {{Customer name}}
          - email: {{Customer email}}
          - summary: {{Customer message summary within 20-30 words}}
          - urgency: {{Customer urgency level, it should return as number data-type, value should lies between 0-1. 0=lower level,0.5=moderated level, 1=highest level}}
          - satisfaction_level: {{Customer satisfaction level,it should return as number data-type, value should lies between 0-1}}
        {format_instructions}
        `,
      ],
      ["human", "Extract information from\n\n{customer_message}"],
    ]);

    // decleare Json Output parser
    const parser = new JsonOutputParser();

    // Initiate a Schema
    const schema = z.object({
      name: z.string().trim().min(1, "Customer name cannot be empty"),
      email: z.string().trim().min(1, "Customer email cannot be empty"),
      summary: z
        .string()
        .trim()
        .min(1, "Customer message summary cannot be empty"),
      urgency: z
        .number()
        .min(0, "Urgency value cannot be negative")
        .max(1, "Urgency value cannot greater than 1"),
      satisfaction_level: z
        .number()
        .min(0, "Satisfaction level value cannot be negative")
        .max(1, "Satisfaction level value cannot greater than 1"),
    });

    // create a RunnableLamda custom function to validate LLM response
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;

      if (!isSuccess) {
        console.log("\n❌ Schema validation error...\n");
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}\n`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log("✅ Schema validation successfull");
      return validate?.data;
    });

    // create chain prompt → LLM → JsonOutputParser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);

    // invoke LLM & get response
    // const response = await chain.invoke({
    //   format_instructions: parser.getFormatInstructions(),
    //   customer_message: customerMessage,
    // });
    const response = await callLLMWithRetresMechanisum(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        customer_message: customerMessage,
      },
      3,
      "pattern_02",
      "customer_message"
    );

    console.log("✅ RESPONSE:", response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_02();

/**
 * Question 3 — Streaming with JSON Output Parser
 * Problem Description: Your team wants to stream JSON output from the LLM for a faster user experience, but still wants a fully parsed JavaScript object at the end of the stream.
 */
async function pattern_03() {
  console.time("# Execution Time");
  console.log("\n=== Question-3: Streaming with JSON Output Parser ===\n");
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a creative character generator. Return the result as JSON object
        - No explaination
        - No code fences
        - No markdown
        Resurn JSON object must contain fields like:
        - name: {{Name of the user}}
        - email: {{email of the user}}
        - phone: {{phone number of the user(maintain indian standard format, example: +91-6291265908)}}
        - country: {{country user belongs from}}
        - city: {{city user belongs from}}
        {format_instructions}
        `,
      ],
      ["human", "Generate a random user"],
    ]);

    // decleare Json Output parser
    const parser = new JsonOutputParser();

    // create chain prompt → LLM → JsonOutputParser
    const chain = prompt.pipe(model).pipe(parser);

    // invoke LLM
    const events = await chain.streamEvents(
      { format_instructions: parser.getFormatInstructions() },
      { version: "v2" }
    );

    let finalResult = null;
    for await (const event of events) {
      const { event: eventname, data } = event;
      if (eventname === "on_parser_stream" && data) {
        console.log("✅ Parsed Data:", data);
        if (Object.keys(data).length === 0) continue;
        finalResult = data?.chunk;
      }
    }
    if (!finalResult) throw new Error("Final value is null");
    const keyLists = ["name", "email", "phone", "country", "city"];
    const missedKeys = keyLists.filter((key) => !(key in finalResult));
    if (missedKeys.length > 0)
      throw new Error(`Missing fields: ${missedKeys.join(", ")}`);

    console.log("✅ FINAL RESPONSE:", finalResult);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_03();

/**
 * Question 4 — Structured Schema Validation with Zod + JSON Parser
 * Problem Description: You're building a resume parser. The LLM output must conform to a strict schema — any missing or wrongly typed field should throw a validation error before the data reaches your application logic.
 */
async function pattern_04() {
  console.time("# Execution Time");
  const validResume = `John Doe Email: john.doe@example.com
    Professional Summary: Experienced software engineer with 8 years in the industry.
    Skills:
      - JavaScript, TypeScript, Node.js
      - React, Next.js
      - PostgreSQL, MongoDB
      - Docker, Kubernetes
    Experience:
      - Senior Engineer at TechCorp (2019 - Present)
      - Junior Developer at StartupXYZ (2016 - 2019)
  `;
  const invalidResume = `Jane Smith Email: not-a-valid-email
  Skills: (none listed)
  Experience: Fresh graduate, no experience yet.
  `;

  console.log(
    "\n=== Question-4: Structured Schema Validation with Zod + JSON Parser ===\n"
  );
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a candidate extractor. Extract candidate information and return valid JSON Object no explaination, no code fences, no markdown.
        RULES:
          - Never hallucinate value
          - If value not found strictly return "", neven return "Unknown", "Invalid" or "Null".
        Return JSON object should contain following fields:
          - name: {{Candidate name}}
          - email: {{Candidate email}}
          - summary: {{Candidate profile summary within 20-30 words}}
          - skills: {{Lists of candidate skills}}
          - experience: {{Candidate total experience. It should be a number data-type}}
        {format_instructions}`,
      ],
      ["human", "Extract candidate information from\n\n{resume_details}"],
    ]);

    // decleare Json Output Parser
    const parser = new JsonOutputParser();

    // create a zod schema
    const schema = z.object({
      name: z.string().trim().min(1, "Candidate name cannot be empty"),
      email: z.string().trim().min(1, "Candidate email cannot be empty"),
      summary: z
        .string()
        .trim()
        .min(1, "Candidate profile summary cannot be empty"),
      skills: z.array(z.string()).min(1, "Minimum 1 skill required"),
      experience: z.number().min(1, "Minimum 1 year of experience is required"),
    });

    // create a custom function to validate LLM response
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;

      if (!isSuccess && error) {
        console.log(`❌ Schema validation error`);
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      return validate?.data;
    });
    // create chain prompt → LLM → JsonOutputParser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);
    // const response = await chain.invoke({
    //   format_instructions: parser.getFormatInstructions(),
    //   // resume_details: invalidResume,
    //   resume_details: validResume,
    // });

    const response = await callLLMWithRetresMechanisum(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        resume_details: invalidResume,
        // resume_details: validResume,
      },
      3,
      "pattern_04",
      "resume_details"
    );
    console.log("✅RESPONSE:", response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_04();

/**
 * Question 5 — Multi-Step Chain with Intermediate JSON Parsing
 * Problem Description: You need a two-step pipeline: the first LLM call extracts raw entities from text, and the second LLM call enriches those entities with additional information — using the JSON output of the first step as input.
 */

// APPROACH-1: Basic implementation of multiple pipeline without retries
async function withoutRetries(travelData) {
  try {
    // *************
    // PINELINE-1
    // *************
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a travel extractor expert. Extract travel information from data and return valid JSON object no markdown, no code fences and no explanation. 
        RULES:
          - Do not hallucinate values
          - If any one of the value is not clear then return "", don't return "Invalid, null or "Unknown"
        Return JSON object must conatin fields:
          - city: {{Name of the city user visit}}
          - country: {{Name of the country}}
          {format_instructions}`,
      ],
      ["human", "Extract trave data from\n\n{travel_data}"],
    ]);
    // create a zod schema to validate LLM response
    const schema = z.object({
      city: z.string().trim().min(1, "City name cannot be empty"),
      country: z.string().trim().min(1, "Coutry name cannot be empty"),
    });
    // create a custom RunnableLamda function to validate response
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;
      if (!isSuccess) {
        console.log(`❌ Schema validation failed in PIPELINE-1`);
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log(`✅ Schema validation successfull in pipeline-1`);
      return validate?.data;
    });
    // decleare a Json output parser
    const parser = new JsonOutputParser();
    // create a chain
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);
    // invoke Chain to call LLM
    const response_1 = await chain.withRetry({ stopAfterAttempt: 2 }).invoke({
      format_instructions: parser.getFormatInstructions(),
      travel_data: travelData,
    });
    console.log("*** ✅ PIPELINE-1 successfull ***");
    console.log(response_1);
    /**
     *
     */
    // *************
    // PINELINE-2
    // *************
    // create a prompt
    const enrichPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a travel enrichment expert. You will receive a JSON with city and country as input. Use YOUR OWN KNOWLEDGE to enrich it with additional details.
        Do not rely only on the input data — the input just tells you WHICH city to enrich.
        Return JSON object must conatin fields:
          - city: {{Name of the city user visit return as string}}
          - country: {{Name of the country return as string}}
          - total_population: {{Total population of city return as number}}
          - currency: {{Currency name Ex. USD, RS etc}}
          - visit_places: {{List of tourist places, should return array of string}}
          - fun_fact: {{Fun fact about the city}}
          - best_time_visit: {{Best time to visit the city}}
          {format_instructions}`,
      ],
      ["human", "Extract travel data from\n\n{travel_data}"],
    ]);
    // create a zod schema to validate LLM response
    const enrichSchema = z.object({
      city: z.string().trim().min(1, "City name cannot be empty"),
      country: z.string().trim().min(1, "Coutry name cannot be empty"),
      total_population: z.number().min(1, "Total population cannot be empty"),
      currency: z.string().trim().min(1, "Country currency cannot be empty"),
      visit_places: z
        .array(z.string())
        .min(1, "Popular tourist places cannot be empty"),
      fun_fact: z
        .string()
        .trim()
        .min(1, "Fun fact about the city cannot be empty"),
      best_time_visit: z
        .string()
        .trim()
        .min(1, "Best time to visit cannot be empty"),
    });
    // create a custom RunnableLamda function to validate response
    const enrichValidateLLMResponse = RunnableLambda.from((input) => {
      // console.log(input);
      const validate = enrichSchema.safeParse(input);
      const { success: isSuccess, error } = validate;
      if (!isSuccess) {
        console.log(`❌ Schema validation failed in PIPELINE-2`);
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log(`✅ Schema validation successfull in pipeline-2`);
      return validate?.data;
    });
    // decleare a Json output parser
    const enrichParser = new JsonOutputParser();
    // create a chain
    const enrichChain = enrichPrompt
      .pipe(model)
      .pipe(enrichParser)
      .pipe(enrichValidateLLMResponse);
    // invoke Chain to call LLM
    const result = await enrichChain.invoke({
      format_instructions: enrichParser.getFormatInstructions(),
      travel_data: JSON.stringify(response_1, null, 2),
    });
    return result;
  } catch (error) {
    throw new Error(`${error.message}`);
  }
}

async function retryLLMCall(chain, input, retries, label) {
  let lastError = null;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const enrichInput = lastError
        ? {
            ...input,
            travel_data:
              input.travel_data +
              `\n\n[PREVIOUS ATTEMPT FAILED WITH ERROR]: ${lastError}. Please fix and retry.`,
          }
        : input;
      const result = await chain(enrichInput);
      if (attempt > 1)
        console.log(`✅ Succeeded on attempt ${attempt} for ${label}`);
      return result;
    } catch (error) {
      lastError = error.message;
      console.warn(
        `⚠️  Attempt ${attempt}/${retries} failed for ${label}: ${error.message}`
      );
      if (attempt < retries) {
        const delay = Math.pow(2, attempt) * 100;
        console.log(`⏳ Retrying in ${delay}ms...`);
        await new Promise((res) => setTimeout(res, delay));
      }
    }
  }
  throw new Error(
    `❌ All ${retries} attempts failed for [${label}]. Last error: ${lastError}`
  );
}
// APPROACH-2: Advance implementation of multiple pipeline with retries & Combine pipeline
async function withRetries(travelData) {
  try {
    // ***************
    // PIPELINE-1
    // ***************
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a travel extractor expert. Extract travel information from data and return valid JSON object no markdown, no code fences and no explanation. 
        RULES:
          - Do not hallucinate values
          - If any one of the value is not clear then return "", don't return "Invalid, null or "Unknown"
        Return JSON object must conatin fields:
          - city: {{Name of the city user visit}}
          - country: {{Name of the country}}
          {format_instructions}`,
      ],
      ["human", "Extract trave data from\n\n{travel_data}"],
    ]);
    // create a zod schema to validate LLM response
    const schema = z.object({
      city: z.string().trim().min(1, "City name cannot be empty"),
      country: z.string().trim().min(1, "Coutry name cannot be empty"),
    });
    // create a Runnable Lamda custom function to validate the resposne
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;
      if (!isSuccess) {
        console.log("❌ Schema validation failed in PIPELINE-1");
        const issues = error.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      return validate?.data;
    });
    // initiate a Json Output parser
    const parser = new JsonOutputParser();
    // create a pipeline
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);
    // invoke LLM with retries mechanisum
    const response_1 = await retryLLMCall(
      (input) => chain.invoke(input),
      {
        format_instructions: parser.getFormatInstructions(),
        travel_data: travelData,
      },
      3,
      "pipeline-1"
    );
    console.log("✅ Response from PIPELINE-1:", response_1);
    // ***************
    // PIPELINE-2
    // ***************
    const enrichPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a travel enrichment expert. You will receive a JSON with city and country as input. Use YOUR OWN KNOWLEDGE to enrich it with additional details.
        Do not rely only on the input data — the input just tells you WHICH city to enrich.
        Return JSON object must conatin fields:
          - city: {{Name of the city user visit return as string}}
          - country: {{Name of the country return as string}}
          - total_population: {{Total population of city return as number}}
          - currency: {{Currency name Ex. USD, RS etc}}
          - visit_places: {{List of tourist places, should return array of string}}
          - fun_fact: {{Fun fact about the city}}
          - best_time_visit: {{Best time to visit the city}}
          {format_instructions}`,
      ],
      ["human", "Extract travel data from\n\n{travel_data}"],
    ]);
    // create a zod schema to validate LLM response
    const enrichSchema = z.object({
      city: z.string().trim().min(1, "City name cannot be empty"),
      country: z.string().trim().min(1, "Coutry name cannot be empty"),
      total_population: z.number().min(1, "Total population cannot be empty"),
      currency: z.string().trim().min(1, "Country currency cannot be empty"),
      visit_places: z
        .array(z.string())
        .min(1, "Popular tourist places cannot be empty"),
      fun_fact: z
        .string()
        .trim()
        .min(1, "Fun fact about the city cannot be empty"),
      best_time_visit: z
        .string()
        .trim()
        .min(1, "Best time to visit cannot be empty"),
    });
    // create a custom RunnableLamda function to validate response
    const enrichValidateLLMResponse = RunnableLambda.from((input) => {
      // console.log(input);
      const validate = enrichSchema.safeParse(input);
      const { success: isSuccess, error } = validate;
      if (!isSuccess) {
        console.log(`❌ Schema validation failed in PIPELINE-2`);
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log(`✅ Schema validation successfull in pipeline-2`);
      return validate?.data;
    });
    // decleare a Json output parser
    const enrichParser = new JsonOutputParser();
    // create a chain
    const enrichChain = enrichPrompt
      .pipe(model)
      .pipe(enrichParser)
      .pipe(enrichValidateLLMResponse);

    const response = await retryLLMCall(
      (input) => enrichChain.invoke(input),
      {
        format_instructions: enrichParser.getFormatInstructions(),
        travel_data: JSON.stringify(response_1, null, 2),
      },
      3,
      "pipeline-2"
    );
    console.log("*".repeat(60), "\n");
    console.log("✅✅ Final result:", response);
  } catch (error) {
    throw new Error(`${error.message}`);
  }
}

async function pattern_05() {
  console.time("# Execution Time");
  const travelBlog = `Last summer, I had the most incredible experience visiting Kyoto, Japan. The ancient temples, the traditional tea houses, and the stunning bamboo groves left me completely speechless. I highly recommend visiting during cherry blossom season — it's truly magical!`;

  console.log(
    "\n=== Question-5: Multi-Step Chain with Intermediate JSON Parsing ===\n"
  );
  try {
    /**
     * APPROACH-1: Basic implementation of multiple pipeline without retries
     */
    // const response_1 = await withoutRetries(travelBlog);
    // console.log(response_1);
    /**
     * APPROACH-2: Advance implementation of multiple pipeline with retries & Combine pipeline
     */
    const response = await withRetries(travelBlog);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_05();

/**
 * Question 6 — Error Handling and Fallback for Malformed JSON
 * Problem Description: In production, LLMs sometimes return malformed JSON (e.g., with trailing commas, unquoted keys, or markdown fences). Your pipeline must be resilient to this.
 */
async function pattern_06() {
  console.time("# Execution Time");
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}

/**
 * Question 7 — Batch Processing with JSON Output Parser
 * Problem Description: You need to process a list of 5 raw customer feedback strings in parallel and convert each one into a structured JSON sentiment report.
 */
async function pattern_07() {
  console.time("# Execution Time");
  console.log("=== Question-7: Batch Processing with JSON Output Parser ===");
  const feedbacks = [
    "I absolutely love this product! It exceeded all my expectations. The quality is outstanding and delivery was super fast.",
    "Terrible experience. The item arrived broken and customer support was completely unhelpful. Never buying again.",
    "It's okay, nothing special. Does what it's supposed to do but the price feels a bit high for what you get.",
    "Amazing value for money! Works perfectly and the build quality is much better than I expected at this price point.",
    "Very disappointed. The description was misleading and the product looks nothing like the photos. Requesting a refund.",
  ];
  try {
    // create a prompt to extract JSON object
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a customer extractor. Extract customer review from customer message and return JSON object with no code fences, no markdown, no explaination. 
        RULES:
          - Never halluccinate values
        And return JSON object must contain these fields:
          - sentiment: {{ Sentiment should be string like "POSITIVE", "NEGATIVE" or "NEUTRAL" }}
          - score: {{ number between -1.0 (most negative) and 1.0 (most positive) }}
          - summary: {{ short summary of the feedback within 15 words  }}
          - key_issues: {{ array of strings listing main complaints (empty array [] if none) }}
          - key_positives: {{ array of strings listing main praises (empty array [] if none) }}
          - would_recommend: {{ boolean }}
          {format_instructions}`,
      ],
      ["human", "Extract information from\n\n{customer_review}"],
    ]);
    // create a zod schema to validate LLM response
    const schema = z.object({
      sentiment: z.string().trim().min(1, "Setiment cannot be empty"),
      score: z.number().describe("Customer review score"),
      summary: z.string().trim().min(1, "Summary cannot be empty"),
      key_issues: z.array(z.string()),
      key_positives: z.array(z.string()),
      would_recommend: z.boolean(),
    });
    // create a helper RunnableLamda function to check response type
    const validateLLMResponse = RunnableLambda.from((input) => {
      const validate = schema.safeParse(input);
      const { success: isSuccess, error } = validate;

      if (!isSuccess) {
        console.log(`❌ Schema validation error`);
        const issues = error?.issues ?? [];
        const errorMsg = issues.map(
          (issue, idx) => `${idx + 1}. ${issue.message}`
        );
        throw new Error(`${errorMsg}`);
      }
      console.log(`✅ Schema validation successfull`);
      return validate?.data;
    });
    // create a Json output parser
    const parser = new JsonOutputParser();
    // create a chain prompt → LLM → JsonOutputParser
    const chain = prompt.pipe(model).pipe(parser).pipe(validateLLMResponse);
    // create batch inputs
    const batchInputs = feedbacks.map((feedback) => ({
      format_instructions: parser.getFormatInstructions(),
      customer_review: feedback,
    }));
    // chain batch invokation
    const results = await chain.batch(batchInputs, { maxConcurrency: 3 });

    console.log(results);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
// pattern_07();

/**
 * Question 8 — Dynamic JSON Schema from User Input
 * Problem Description: You're building a no-code data extraction tool where users define their own output schema at runtime by providing field names and types. The parser must adapt dynamically to whatever schema the user specifies.
 */
async function pattern_08() {
  console.time("# Execution Time");
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}

/**
 * Question 9 — JSON Repair & Retry Chain with Memory
 * Problem Description: You are building a conversational data extraction assistant. If the LLM's JSON output is invalid, the system should retry automatically by sending the malformed output back to the LLM with an instruction to fix it — up to 2 retry attempts — before giving up.
 */
async function pattern_09() {
  console.time("# Execution Time");
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}

/**
 * Question 10 — Parallel Multi-Schema Extraction with Aggregation
 * Problem Description: You're building a document intelligence pipeline where a single input document must be analyzed from multiple angles simultaneously — each producing a different JSON schema — and the results must be merged into one unified report object.
 */
async function pattern_10() {
  console.time("# Execution Time");
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(0);
  }
}
