/** @format */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { model } from "../../configs/model.config.js";
import { JsonOutputParser } from "@langchain/core/output_parsers";

/**
 * ✅ EXAMPLE 1 — Resume Parsing
 * Description: Extract structured candidate information from a free-text résumé so it can be stored in an ATS (Applicant Tracking System).
 *
 */
async function pattern_01() {
  console.time("# Execution Time");
  try {
    console.log(`=== Resume Parsing ===\n`);
    const resumeText = `
    Jane Doe | jane@example.com | +1-555-0100
    Skills: Python, Machine Learning, SQL, Docker, AWS
    Experience:
      - Senior Data Scientist @ Acme Corp (2020-2024)
      - Data Analyst @ Beta Inc (2017-2020)
    Education:
      - M.Sc. Computer Science, MIT, 2017
      - B.Sc. Mathematics, UCLA, 2015
  `;

    // create a resume parser prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a résumé parser. Extract ALL fields from the résumé and return ONLY valid JSON.
{format_instructions}`,
      ],
      ["human", "Parse this résumé:\n\n{resume} "],
    ]);

    // initiate json parser
    const parser = new JsonOutputParser();

    // create a chain
    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instructions: parser.getFormatInstructions(),
      resume: resumeText,
    });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
  }
}
// pattern_01();

/**
 * ✅ EXAMPLE 2 — Article Metadata Extraction
 * Automatically tag blog articles with SEO-friendly metadata (title, summary, keywords, category, reading-time estimate).
 */
async function pattern_02() {
  console.time("# Execution Time");
  try {
    const article = `
    Quantum computing is rapidly transitioning from the research lab to
    real-world applications. Companies like IBM, Google, and IonQ are
    racing to achieve quantum advantage — a point where quantum processors
    outperform classical computers on commercially valuable tasks.
    Recent breakthroughs in error correction have pushed qubit fidelity
    above 99.9%, making fault-tolerant quantum computing a near-term goal.
    Industries such as pharmaceuticals, finance, and logistics stand to
    gain the most from this technology.
  `;

    // create a prompt for Article Metadata Extraction
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a article information extractor, extract valid json format information like title, summary, keyword, category and reading-time estimate  from article {format_instruction}`,
      ],
      ["human", "Extract information from:\n\n{article}"],
    ]);

    // initiate json output parser
    const parser = new JsonOutputParser();

    // create a chain
    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instruction: parser.getFormatInstructions(),
      article,
    });
    console.log(response);
  } catch (error) {
    console.log("Error: ", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
  }
}
// pattern_02();

/**
 * ✅ EXAMPLE 3 — Structured Product Information
 * Convert a raw, unstructured product description (from a vendor email or scraped page) into a clean catalogue record.
 */
async function pattern_03() {
  console.time("# Execution Time");
  try {
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

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a product information extractor, extract information in valid json format. should contain these mention fields: productName, sku, price(should be as number datatype), currency, category, specifications, inStock(should be boolean), tags. {format_instruction}",
      ],
      ["human", "Extract information from: \n\n{description}"],
    ]);

    const parser = new JsonOutputParser();
    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instruction: parser.getFormatInstructions(),
      description: rawDescription,
    });
    console.log(response);
  } catch (error) {
    console.log("Error: ", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
  }
}
// pattern_03();

/**
 * ✅ EXAMPLE 4 — Structured Summary Generation
 * Summarise a meeting transcript into a structured briefing note suitable for executives who missed the meeting.
 *  * {
 *   meetingTitle: string,
 *   date: string,
 *   attendees: string[],
 *   keyDecisions: string[],
 *   actionItems: [{ owner: string, task: string, dueDate: string }],
 *   nextMeetingDate: string | null
 * }
 */
async function pattern_04() {
  console.time("# Execution Time");
  try {
    const transcript = `
    Meeting: Q3 Product Roadmap Review — 2024-09-15
    Attendees: Sarah (PM), Tom (Eng Lead), Priya (Design), Marcus (Sales)

    Sarah opened by reviewing Q2 metrics. Tom confirmed the mobile
    redesign will ship by Oct 1st. Priya presented three new dashboard
    mockups — the team voted for Option B. Marcus requested a CRM
    integration by end of Q4; Tom estimated 6 weeks of effort.
    Action items:
      - Tom: kick off CRM spike by Sep 20
      - Priya: finalise Option B screens by Sep 22
      - Sarah: update roadmap deck and share with leadership by Sep 17
    Next meeting: Oct 1st post-launch retro.
  `;

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a meeting summarising expert. Transcript the information into valid json format and should contain meetingTitle, date, attendees, keyDecisions, actionItems, nextMeetingDate. format: {format_instruction}",
      ],
      ["human", "Trnascript this meeting data: \n\n {transcript}"],
    ]);

    const parser = new JsonOutputParser();

    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instruction: parser.getFormatInstructions(),
      transcript: transcript,
    });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.time("# Execution Time");
    process.exit(1);
  }
}
// pattern_04();

/**
 * ✅ EXAMPLE 5 — User Input → Form Data (Lead Capture)
 * A chatbot collects lead information from a conversational message and normalises it into a CRM-ready JSON payload.
 */
async function pattern_05() {
  console.time("# Execution Time");
  try {
    const userMessage = `
    Hi! I'm Alex johnson, alex.j@bigcorp.com, my number is 555 867 5309.
    BigCorp has about 5,000 employees. We're interested in your enterprise
    analytics and API products. Best to reach me by email.
  `;
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are lead capture expert. Extract user data like user name, email, phone number(should be as number & normalize the phone number like 555-867-5309), company name, with company size(as number), user interests, mode of communication from user given information and return the information into valid json format {format_instruction}",
      ],
      ["human", "Extract information\n\n{user_message}"],
    ]);

    const parse = new JsonOutputParser();

    const response = await prompt.pipe(model).pipe(parse).invoke({
      format_instruction: parse.getFormatInstructions(),
      user_message: userMessage,
    });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
// pattern_05();

/**
 * ✅ EXAMPLE 6 — Task Planning Output
 * Break a high-level project goal into a structured sprint plan that a project management tool (Jira, Linear) can import directly.
 */
async function pattern_06() {
  console.time("# Execution Time");
  try {
    const goal = `
    Launch a B2B SaaS onboarding redesign that reduces time-to-value
    from 14 days to 3 days within one quarter.
  `;

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are an agile project planner. Given a project goal, output ONLY a JSON sprint plan with:projectName, g oal, estimatedWeeks, epics (each with epicName, priority, tasks (each with taskName, assignee role, storyPoints 1-13, dependsOn array)). Keep it to 3 epics and 3 tasks per epic. {format_instructions}`,
      ],
      ["human", "Project goal: {goal}"],
    ]);

    const parser = new JsonOutputParser();

    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instructions: parser.getFormatInstructions(),
      goal,
    });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
// pattern_06();

/**
 * ✅ EXAMPLE 7 — Sentiment & Intent Analysis (API-Ready Response)
 * Analyse a customer support ticket and return a structured triage object that a support platform's API can act on immediately
 */
async function pattern_07() {
  console.time("# Execution Time");
  try {
    const ticket = {
      id: "TKT-88421",
      body: `I've been waiting THREE WEEKS for my refund and nobody has
replied to my last four emails. This is absolutely unacceptable.
I'm about to dispute the charge with my bank if I don't hear back TODAY.`,
    };

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a support triage AI. Extract information like ticket id, sentiment(positive, neutral, negative, angry), urgency, intent, department, confidence (0-1), mode of communication, message, from ticket details, return valid JSON format {format_instructions}",
      ],
      ["human", "Extract information from\n\n {ticket_details}"],
    ]);

    const parser = new JsonOutputParser();

    const response = await prompt.pipe(model).pipe(parser).invoke({
      format_instructions: parser.getFormatInstructions(),
      ticket_details: ticket,
    });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
// pattern_07();

/**
 * ✅ EXAMPLE 8 — Named Entity Recognition (NER)
 * Extract named entities from a news snippet for knowledge-graph
 * ingestion — people, organisations, locations, dates, and monetary
 * figures, each typed and linked.
 */
async function pattern_08() {
  console.time("# Execution Time");
  try {
    const newsSnippet = `
    On March 3 2025, Elon Musk's xAI announced a $6 billion Series B
    funding round led by Sequoia Capital and Andreessen Horowitz.
    The deal values the San Francisco-based startup at $50 billion.
    The funds will be used to expand xAI's Memphis data centre.
  `;
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are an NER system.
Extract all named entities and return ONLY JSON:
{{ sourceText: string, entities: array of {{ text, type (PERSON|ORG|LOC|DATE|MONEY), context }} }}.
{format_instructions}`,
      ],
      ["human", "{text}"],
    ]);

    const response = await prompt
      .pipe(model)
      .pipe(new JsonOutputParser())
      .invoke({
        format_instructions: new JsonOutputParser().getFormatInstructions(),
        text: newsSnippet,
      });

    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
// pattern_08();

/**
 * ✅ EXAMPLE 9 — Multi-Language Invoice Data Extraction
 * Parse a foreign-language invoice (French) into a normalised
 * accounting JSON record with all amounts converted to a standard
 * structure.
 */
async function pattern_09() {
  console.time("# Execution Time");
  try {
    const frenchInvoice = `
    FACTURE N° 2025-0342
    Date d'émission: 10 février 2025
    Date d'échéance: 10 mars 2025

    Fournisseur: Dupont Informatique SARL
    Adresse: 12 Rue de Rivoli, 75001 Paris, France
    N° TVA: FR 40 123456789

    Désignation                  Qté   P.U. HT   Total HT
    Licence logiciel Enterprise   3     800,00 €  2 400,00 €
    Support technique (mensuel)   1     500,00 €    500,00 €

    Sous-total HT:  2 900,00 €
    TVA (20%):        580,00 €
    TOTAL TTC:      3 480,00 €
  `;

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a multi language accounts-payable data extractor. Parse the invoice (any language) into English-keyed and return valid JSON format:invoiceNumber, issueDate (ISO 8601), dueDate (ISO 8601), vendor (name, address, vatId), lineItems (description, qty, unitPrice, total), subtotal, taxRate, taxAmount, totalAmount, currency (ISO 4217). {format_instructions}`,
      ],
      ["human", "Extract information from\n\n${invoice_data}"],
    ]);

    const response = await prompt
      .pipe(model)
      .pipe(new JsonOutputParser())
      .invoke({
        format_instructions: new JsonOutputParser().getFormatInstructions(),
        invoice_data: frenchInvoice,
      });
    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
// pattern_09();

/**
 * EXAMPLE 10 — Dynamic Quiz / Assessment Generator
 * Generate a multiple-choice quiz on any topic with correct answers
 * and explanations — ready to be consumed by an e-learning platform's
 * question-bank API.
 */
async function pattern_10() {
  console.time("# Execution Time");
  try {
    const params = {
      topic: "JavaScript Promises and async/await",
      difficulty: "intermediate",
      count: 3,
    };

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a quiz author for a developer e-learning platform. Generate a multiple-choice quiz and resturn into valid JSON format, questions should contain those fields like topic, difficulty, questions, id(number), options:A/B/C/D, correct_answer, explaination. {format_instructions}`,
      ],
      ["human", "Generate questions\n\n{topic_details}"],
    ]);

    const response = await prompt
      .pipe(model)
      .pipe(new JsonOutputParser())
      .invoke({
        format_instructions: new JsonOutputParser().getFormatInstructions(),
        topic_details: params,
      });

    console.log(response);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("# Execution Time");
    process.exit(1);
  }
}
pattern_10();
